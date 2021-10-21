#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 19.10.21
"""

import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.make_html_page import make_html_page
from myutils.project_utils import *
from x_utils.vpf_sevices import get_vpf_service
from root_dir import DATA_DIR


class ModelEvaluator(object):
    """
    评估模型
    """
    def __init__(self):
        pass

    @staticmethod
    def prepare_dataset():
        folder_path = os.path.join(DATA_DIR, "datasets", "text_line_dataset_c4_20211013_files")
        file1 = os.path.join(folder_path, "train_000.txt")
        file2 = os.path.join(folder_path, "train_001.txt")
        file3 = os.path.join(folder_path, "train_002.txt")
        file4 = os.path.join(folder_path, "train_003.txt")
        file_list = [file1, file2, file3, file4]
        val_file = os.path.join(DATA_DIR, "files_text_line_v2", "val_file_4k_v2.txt")
        num = 1000
        random.seed(47)
        for file_idx, file in enumerate(file_list):
            data_lines = read_file(file)
            random.shuffle(data_lines)
            data_lines = data_lines[:num]
            out_lines = ["{}\t{}".format(data_line, file_idx) for data_line in data_lines]
            write_list_to_file(val_file, out_lines)
        print('[Info] 处理完成: {}, 样本数: {}'.format(val_file, len(read_file(val_file))))

    @staticmethod
    def predict_img_url(img_url):
        """
        预测图像url
        """
        res_dict = get_vpf_service(img_url, service_name="LvQAecdZrrxkLFs6QiUXsF")
        p_list = res_dict["data"]["prob_list"]
        return p_list

    @staticmethod
    def process_line(data_idx, data_line, out_file):
        url, label = data_line.split("\t")
        p_list = ModelEvaluator.predict_img_url(url)
        p_str_list = [str(round(pr, 4)) for pr in p_list]
        write_line(out_file, "\t".join([url, label] + p_str_list))
        if data_idx % 10 == 0:
            print('[Info] data_idx: {}'.format(data_idx))

    @staticmethod
    def predict_dataset():
        val_file = os.path.join(DATA_DIR, "files_text_line_v2", "val_file_4k_v2.txt")
        out_file = os.path.join(DATA_DIR, "files_text_line_v2", "val_file_4k_v2_out.{}.txt".format(get_current_time_str()))
        print('[Info] 评估文件: {}'.format(val_file))
        data_lines = read_file(val_file)
        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(data_lines):
            # ModelEvaluator.process_line(data_idx, data_line, out_file)
            pool.apply_async(ModelEvaluator.process_line, (data_idx, data_line, out_file))
        pool.close()
        pool.join()
        print('[Info] 处理完成: {}, 样本数: {}'.format(out_file, len(read_file(out_file))))

    @staticmethod
    def get_results_data():
        out_file = os.path.join(DATA_DIR, "files_text_line_v2", "val_file_4k_v2_out.20211021160418.txt")
        print('[Info] 测试结果: {}'.format(out_file))
        data_lines = read_file(out_file)
        items_list = []
        for data_line in data_lines:
            items = data_line.split("\t")
            items_list.append(items)
        print('[Info] 测试样本数: {}'.format(len(items_list)))
        return items_list

    @staticmethod
    def confusion_matrix(items_list, label_str_list):
        print('[Info] ' + "-" * 100)
        print('[Info] 混淆矩阵')
        label_dict = collections.defaultdict(list)
        for items in items_list:
            url = items[0]
            gt = int(items[1])
            prob_list = [float(i) for i in items[2:]]
            pl = np.argmax(prob_list)
            label_dict[gt].append(pl)

        for label in label_dict.keys():
            num_dict = list_2_numdict(label_dict[label])
            num_dict = sort_dict_by_key(num_dict)
            print('[Info] {}'.format(label_str_list[label]))
            print(["{}:{}%".format(label_str_list[items[0]], items[1]/10) for items in num_dict])

        label_dict = dict()
        for items in items_list:
            url = items[0]
            gt = int(items[1])
            prob_list = [float(i) for i in items[2:]]
            pl = int(np.argmax(prob_list))
            if pl not in label_dict.keys():
                label_dict[pl] = collections.defaultdict(list)
            label_dict[pl][gt].append([url, prob_list[gt], prob_list[pl]])  # 预测label

        out_dir = os.path.join(DATA_DIR, "results_text_line_v2")
        mkdir_if_not_exist(out_dir)
        for pl_ in label_dict.keys():  # 预测label
            gt_dict = label_dict[pl_]
            gt_list_dict = dict()
            for gt_ in gt_dict.keys():
                url_prob_list = gt_dict[gt_]
                url_list, gt_list, pl_list = [], [], []
                for url_prob in url_prob_list:
                    url, gt_prob, pl_prob = url_prob
                    gt_list.append(gt_prob)
                    pl_list.append(pl_prob)
                    url_list.append(url)
                pl_list, gt_list, url_list = sort_three_list(pl_list, gt_list, url_list, reverse=True)
                gt_list_dict[gt_] = [pl_list, gt_list, url_list]

            for gt_ in gt_list_dict.keys():
                gt_str = label_str_list[gt_]
                pl_str = label_str_list[pl_]
                out_html = os.path.join(out_dir, "pl{}_gt{}.html".format(pl_str, gt_str))
                pl_list, gt_list, url_list = gt_list_dict[gt_]
                html_items = [[url, pl_str, pl_prob, gt_str, gt_prob]
                              for gt_prob, pl_prob, url in zip(gt_list, pl_list, url_list)]
                make_html_page(out_html, html_items)

        print('[Info] ' + "-" * 100)

    @staticmethod
    def pr_curves(items_list):
        target_list = []
        target_idx = 0
        positive_num = 0
        for items in items_list:
            url = items[0]
            gt = int(items[1])
            prob_list = [float(i) for i in items[2:]]
            pl = np.argmax(prob_list)
            tar_prob, other_prob = 0, 0
            for prob_idx, prob in enumerate(prob_list):
                if prob_idx == target_idx:
                    tar_prob += prob
                else:
                    other_prob += prob
            tar_label = 0 if gt == target_idx else 1
            positive_num += 1 if tar_label == 0 else 0
            # print('[Info] tar_label: {}, tar_prob: {}, other_prob: {}'.format(tar_label, tar_prob, other_prob))
            target_list.append([tar_label, tar_prob, other_prob])
        print('[Info] 样本数: {}, 正例数: {}'.format(len(target_list), positive_num))
        prob_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        for prob in prob_list:
            recall_num = 0
            precision_num = 0
            x_num = 0
            for items in target_list:
                tar_label = items[0]
                tar_prob = items[1]
                pl = 0 if tar_prob >= prob else 1
                if pl == 0 and tar_label == 0:
                    x_num += 1
                if tar_label == 0:
                    recall_num += 1
                if pl == 0:
                    precision_num += 1
            recall = x_num / recall_num
            precision = x_num / precision_num
            print('[Info] prob: {}, recall: {}'.format(prob, recall))
            print('[Info] prob: {}, precision: {}'.format(prob, precision))

    @staticmethod
    def process_results():
        items_list = ModelEvaluator.get_results_data()
        label_str_list = ["印刷文本", "手写文本", "艺术字", "无文字"]
        ModelEvaluator.confusion_matrix(items_list, label_str_list)  # 计算混淆矩阵
        # ModelEvaluator.pr_curves(items_list)  # 计算混淆矩阵

    @staticmethod
    def process():
        # 第1步，准备数据集
        # ModelEvaluator.prepare_dataset()

        # 第2步，预测数据结果
        # ModelEvaluator.predict_dataset()

        # 第3步，处理测试结果
        ModelEvaluator.process_results()


def main():
    de = ModelEvaluator()
    de.process()


if __name__ == '__main__':
    main()
