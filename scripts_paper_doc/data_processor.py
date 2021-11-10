#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 27.9.21
"""

import os
import sys

import cv2

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myscripts.dataset_utils import generate_dataset_mul
from root_dir import DATA_DIR
from myutils.project_utils import *


class DataProcessor(object):
    """
    数据处理类
    """
    def __init__(self):
        pass

    @staticmethod
    def process_v1():
        file_name = os.path.join(DATA_DIR, "files_paper_doc", "out_labeled_urls.txt")
        out_doc_name = os.path.join(DATA_DIR, "files_paper_doc", "doc_v1.txt")
        out_nat_name = os.path.join(DATA_DIR, "files_paper_doc", "nat_v1.txt")

        print('[Info] 文件: {}'.format(file_name))
        data_lines = read_file(file_name)
        print('[Info] 文本行数: {}'.format(len(data_lines)))
        out_dict = collections.defaultdict(list)
        for data_line in data_lines:
            items = data_line.split("\t")
            url = items[0]
            label = items[1]
            if label == "0":
                out_dict[label].append(url)
            else:
                out_dict["1"].append(url)

        print('[Info] out_dict: {}'.format(out_dict.keys()))
        for label in out_dict.keys():
            if label == "0":
                create_file(out_doc_name)
                write_list_to_file(out_doc_name, out_dict[label])
            else:
                create_file(out_nat_name)
                write_list_to_file(out_nat_name, out_dict[label])


    @staticmethod
    def process_v2():
        file_name = os.path.join(DATA_DIR, "files_paper_doc", "文字检测_识别文字检测数据清洗_.txt")
        out_doc_name = os.path.join(DATA_DIR, "files_paper_doc", "doc_v2.txt")
        out_nat_name = os.path.join(DATA_DIR, "files_paper_doc", "nat_v2.txt")

        print('[Info] 文件: {}'.format(file_name))
        data_lines = read_file(file_name)
        print('[Info] 文本行数: {}'.format(len(data_lines)))
        out_dict = collections.defaultdict(list)
        for data_line in data_lines:
            data_dict = json.loads(data_line)
            url = data_dict["url"]
            label = data_dict["cate"]
            out_dict[label].append(url)

        print('[Info] out_dict: {}'.format(out_dict.keys()))
        for label in out_dict.keys():
            if label == "文档":
                create_file(out_doc_name)
                write_list_to_file(out_doc_name, out_dict[label])
            else:
                create_file(out_nat_name)
                write_list_to_file(out_nat_name, out_dict[label])

    @staticmethod
    def merge():
        out1_doc_name = os.path.join(DATA_DIR, "files_paper_doc", "doc_v1.txt")
        out1_nat_name = os.path.join(DATA_DIR, "files_paper_doc", "nat_v1.txt")
        out2_doc_name = os.path.join(DATA_DIR, "files_paper_doc", "doc_v2.txt")
        out2_nat_name = os.path.join(DATA_DIR, "files_paper_doc", "nat_v2.txt")

        out_doc_name = os.path.join(DATA_DIR, "files_paper_doc", "doc_all.txt")
        out_nat_name = os.path.join(DATA_DIR, "files_paper_doc", "nat_all.txt")

        data1_doc_lines = read_file(out1_doc_name)
        data2_doc_lines = read_file(out2_doc_name)
        data_doc_lines = data1_doc_lines + data2_doc_lines
        print('[Info] doc: {}'.format(len(data_doc_lines)))
        write_list_to_file(out_doc_name, data_doc_lines)

        data1_nat_lines = read_file(out1_nat_name)
        data2_nat_lines = read_file(out2_nat_name)
        data_nat_lines = data1_nat_lines + data2_nat_lines
        print('[Info] nat: {}'.format(len(data_nat_lines)))
        write_list_to_file(out_nat_name, data_nat_lines)

    @staticmethod
    def process_dataset(data_line, label_dir, label_idx, data_idx):
        _, img_bgr = download_url_img(data_line)
        file_path = os.path.join(label_dir, "{}_{}.jpg".format(str(label_idx).zfill(3), str(data_idx).zfill(7)))
        cv2.imwrite(file_path, img_bgr)
        if data_idx % 1000 == 0:
            print("[Info] \t data_idx: {}".format(data_idx))
            time.sleep(10)

    @staticmethod
    def make_dataset():
        """
        构建数据集
        """
        dataset_dir = os.path.join(DATA_DIR, "datasets", "paper_doc_dataset_c2_20211009")

        doc_path = os.path.join(DATA_DIR, "files_paper_doc", "doc_all.txt")
        nat_path = os.path.join(DATA_DIR, "files_paper_doc", "nat_all.txt")

        gap = 20
        num = 60000

        data1_lines = read_file(doc_path)
        train1_data, val1_data = split_train_and_val(data1_lines, gap)
        train1_data = get_fixed_samples(train1_data, num)
        val1_data = get_fixed_samples(val1_data, num // gap)
        print('[Info] train1_data: {}'.format(len(train1_data)))
        print('[Info] val1_data: {}'.format(len(val1_data)))

        data2_lines = read_file(nat_path)
        train2_data, val2_data = split_train_and_val(data2_lines)
        train2_data = get_fixed_samples(train2_data, num)
        val2_data = get_fixed_samples(val2_data, num // gap)
        print('[Info] train2_data: {}'.format(len(train2_data)))
        print('[Info] val2_data: {}'.format(len(val2_data)))

        # 抽象为单独的函数
        generate_dataset_mul(dataset_dir, [train1_data, train2_data], [val1_data, val2_data])

    @staticmethod
    def generate_test_file():
        """
        生成测试文件
        """
        out_file = os.path.join(DATA_DIR, "files_paper_doc", "test_file_1000.txt")
        print('[Info] 输出文件: {}'.format(out_file))
        create_file(out_file)
        file1 = os.path.join(DATA_DIR, "files_paper_doc", "doc_all.txt")
        file2 = os.path.join(DATA_DIR, "files_paper_doc", "nat_all.txt")

        data_lines1 = read_file(file1)
        data_lines2 = read_file(file2)

        random.seed(47)
        random.shuffle(data_lines1)
        random.shuffle(data_lines2)
        data_lines1 = data_lines1[:500]
        data_lines2 = data_lines2[:500]

        out_lines = []
        for data_line in data_lines1:
            out_lines.append("\t".join([data_line, "0"]))
        for data_line in data_lines2:
            out_lines.append("\t".join([data_line, "1"]))

        print('[Info] 输出样本数: {}'.format(len(out_lines)))
        write_list_to_file(out_file, out_lines)
        print('[Info] 写入文件完成: {}'.format(out_file))


def main():
    dp = DataProcessor()
    # dp.make_dataset()
    dp.generate_test_file()


if __name__ == '__main__':
    main()
