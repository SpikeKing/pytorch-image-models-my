#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 29.9.21
"""

import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)


from myutils.make_html_page import make_html_page
from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import DATA_DIR


class Labeled2Dataset(object):
    """
    类别标注文件转换为数据集
    """
    def __init__(self):
        pass

    @staticmethod
    def process_csv_row_v1(row_list):
        data_dict = collections.defaultdict(list)
        for row_idx, row in enumerate(row_list):
            url = eval(row["问题内容"])[0]
            label = eval(row["回答内容"])["radio_1"]
            if not label:
                continue
            data_dict[url].append(label)
        return data_dict

    @staticmethod
    def process_csv_row_v2(row_list, label_dict):
        data_dict = collections.defaultdict(list)
        for row_idx, row in enumerate(row_list):
            url = eval(row["问题内容"])[0]
            pre_label = eval(row["问题内容"])[1]
            radio_1 = eval(row["回答内容"])["radio_1"]
            radio_2 = eval(row["回答内容"])["radio_2"]
            if radio_1 == "0":
                label = pre_label
                data_dict[url].append(label)
            else:
                if not radio_2:
                    continue
                data_dict[url].append(label_dict[int(radio_2)])
        return data_dict

    @staticmethod
    def load_data_file(file_list, label_dict):
        """
        加载数据文件
        """
        row_list = []
        column_names = []
        for file_name in file_list:
            column_names, sub_row_list = read_csv_file(file_name)
            print('[Info] column_names: {}'.format(column_names))
            row_list += sub_row_list
        print('[Info] 样本数: {}'.format(len(row_list)))
        print('[Info] row: {}'.format(row_list[0]))

        data_dict = Labeled2Dataset.process_csv_row_v2(row_list, label_dict)

        right_dict = collections.defaultdict(list)
        for url in data_dict.keys():
            raw_labels = data_dict[url]
            labels = list(set(raw_labels))
            if len(labels) == 1 and len(raw_labels) >= 2:
                label = labels[0]
                if not label:
                    continue
                right_dict[label].append(url)

        print("-" * 200)
        all_num = 0
        for label in right_dict.keys():
            num = len(right_dict[label])
            all_num += num
        print('[Info] 样本总数: {}'.format(all_num))

        for label in right_dict.keys():
            print('[Info] ' + "-" * 50)
            print('[Info] label: {}'.format(label))
            print('[Info] 样本数: {}'.format(len(right_dict[label])))
            print('[Info] 占比: {} %'.format(safe_div(len(right_dict[label]), all_num) * 100))
        print("-" * 200)

        return right_dict

    @staticmethod
    def merge_data_dict(data1_dict, data2_dict, filter_id=0):
        """
        合并数据dict
        """
        res_dict = collections.defaultdict(list)
        for label in data1_dict.keys():
            if int(label) != filter_id:
                urls = data1_dict[label]
                res_dict[label] += urls
        for label in data2_dict.keys():
            urls = data2_dict[label]
            res_dict[label] += urls
        return res_dict

    @staticmethod
    def save_url(idx, url, file_name):
        _, img_bgr = download_url_img(url)
        h, w, _ = img_bgr.shape
        area = h * w
        if area < 200 * 200:
            return
        write_line(file_name, url)
        if idx % 1000 == 0:
            print('[Info] \t处理完成: {}'.format(idx))

    @staticmethod
    def save_data(right_dict, label_format_name, html_name):
        pool = Pool(processes=100)
        for label in right_dict.keys():
            print('[Info] ' + "-" * 50)
            print('[Info] 样本数: {}'.format(len(right_dict[label])))
            label_file = label_format_name.format(label)
            create_file(label_file)
            # write_list_to_file(label_file, right_dict[label])  # 直接写入
            for idx, url in enumerate(right_dict[label]):
                # Labeled2Dataset.save_url(idx, url, label_file)
                pool.apply_async(Labeled2Dataset.save_url, (idx, url, label_file))
        pool.close()
        pool.join()

        count = 100
        items = []
        for label in right_dict.keys():
            urls = right_dict[label]
            random.seed(47)
            random.shuffle(urls)
            urls = urls[:count]
            for url in urls:
                items.append([url, label])

        make_html_page(html_name, items)
        print('[Info] 处理完成: {}'.format(html_name))

    @staticmethod
    def process_v1():
        file1_name = os.path.join(DATA_DIR, "files_text_line", "text_line_6c_file1.csv")
        file2_name = os.path.join(DATA_DIR, "files_text_line", "text_line_6c_file2.csv")
        file3_name = os.path.join(DATA_DIR, "files_text_line", "text_line_6c_file3.csv")
        html_name = os.path.join(DATA_DIR, "files_text_line", "text_line_6c_out.html")
        label_format_name = os.path.join(DATA_DIR, "files_text_line", "text_line_6c_{}.txt")
        label_dict = {0: "印刷公式", 1: "印刷文本", 2: "手写公式", 3: "手写文本", 4: "艺术字", 5: "其他"}

        print('[Info] 处理文件: {}'.format(file1_name))
        print('[Info] 处理文件: {}'.format(file2_name))
        print('[Info] 处理文件: {}'.format(file3_name))

        right_dict = Labeled2Dataset.load_data_file([file1_name, file2_name, file3_name], label_dict)

        Labeled2Dataset.save_data(right_dict, label_format_name, html_name)

    @staticmethod
    def process_v2():
        """
        处理第2批数据，主要明确标注 印刷文字 和 艺术字
        """
        file1_name = os.path.join(DATA_DIR, "files_text_line", "text_line_4c_file1.20211008.csv")
        file2_name = os.path.join(DATA_DIR, "files_text_line", "text_line_4c_file2.20211008.csv")
        file3_name = os.path.join(DATA_DIR, "files_text_line", "text_line_4c_file3.20211008.csv")
        html_name = os.path.join(DATA_DIR, "files_text_line", "text_line_4c_out.20211008.html")
        label_format_name = os.path.join(DATA_DIR, "files_text_line", "text_line_4c_{}.20211008.txt")
        label_dict = {0: "印刷文字", 1: "手写文字", 2: "艺术字", 3: "无文字", 4: "抛弃"}

        print('[Info] 处理文件: {}'.format(file1_name))
        print('[Info] 处理文件: {}'.format(file2_name))
        print('[Info] 处理文件: {}'.format(file3_name))
        right1_dict = Labeled2Dataset.load_data_file([file1_name, file2_name], label_dict)
        right2_dict = Labeled2Dataset.load_data_file([file3_name], label_dict)

        right_dict = Labeled2Dataset.merge_data_dict(right1_dict, right2_dict)

        Labeled2Dataset.save_data(right_dict, label_format_name, html_name)

    @staticmethod
    def process_v3():
        file_path = os.path.join(DATA_DIR, "files_text_line", "text_line_doc_dataset.min100x100.txt")
        out1_path = os.path.join(DATA_DIR, "files_text_line", "text_line_doc_印刷文字.txt")
        out2_path = os.path.join(DATA_DIR, "files_text_line", "text_line_doc_手写文字.txt")
        label_str_dict = {"0": "其他", "1": "印刷公式", "2": "印刷文本", "3": "手写公式", "4": "手写文本", "5": "艺术字"}
        data_lines = read_file(file_path)
        print('[Info] 文件数: {}'.format(len(data_lines)))
        out1_list, out2_list = [], []
        for data_line in data_lines:
            data_dict = json.loads(data_line)
            crop_img_url = data_dict["crop_img_url"]
            # _, img_bgr = download_url_img(crop_img_url)
            # h, w, _ = img_bgr.shape
            # area = h * w
            # print("area: {}, url: {}".format(area, crop_img_url))
            label = data_dict["label"]
            if label == 1 or label == 2:
                out1_list.append(crop_img_url)
            elif label == 3 or label == 4:
                out2_list.append(crop_img_url)
        create_file(out1_path)
        write_list_to_file(out1_path, out1_list)
        create_file(out2_path)
        write_list_to_file(out2_path, out2_list)

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
        folder_path = os.path.join(DATA_DIR, "files_text_line")
        file1_path = os.path.join(folder_path, "text_line_doc_印刷文字.txt")
        file2_path = os.path.join(folder_path, "text_line_doc_手写文字.txt")
        file3_path = os.path.join(folder_path, "text_line_4c_印刷文字.20211008.txt")
        file4_path = os.path.join(folder_path, "text_line_4c_手写文字.20211008.txt")
        file5_path = os.path.join(folder_path, "text_line_4c_艺术字.20211008.txt")
        file6_path = os.path.join(folder_path, "text_line_4c_无文字.20211008.txt")

        data1_lines = read_file(file1_path)
        data1_lines = get_fixed_samples(data1_lines, 100000)

        data2_lines = read_file(file2_path)
        data2_lines = get_fixed_samples(data2_lines, 50000)

        data3_lines = read_file(file3_path)
        data3_lines = get_fixed_samples(data3_lines, 30000)

        data4_lines = read_file(file4_path)
        data4_lines = get_fixed_samples(data4_lines, 5000)

        data5_lines = read_file(file5_path)
        data5_lines = get_fixed_samples(data5_lines, 50000)

        data6_lines = read_file(file6_path)
        data6_lines = get_fixed_samples(data6_lines, 5000)

        data_list = [data1_lines, data2_lines, data3_lines, data4_lines, data5_lines, data6_lines]

        dataset = os.path.join(DATA_DIR, "datasets", "text_line_dataset_20211008")
        mkdir_if_not_exist(dataset)

        pool = Pool(processes=100)
        for label_idx, data_lines in enumerate(data_list):
            label_dir = os.path.join(dataset, "{}".format(str(label_idx).zfill(3)))
            mkdir_if_not_exist(label_dir)
            for data_idx, data_line in enumerate(data_lines):
                # Labeled2Dataset.process_dataset(data_line, label_dir, label_idx, data_idx)
                pool.apply_async(Labeled2Dataset.process_dataset, (data_line, label_dir, label_idx, data_idx))
        pool.close()
        pool.join()
        print('[Info] 数据集处理完成: {}'.format(dataset))


def main():
    l2d = Labeled2Dataset()
    # l2d.process_v1()
    l2d.make_dataset()
    # l2d.filter_labeled_file()


if __name__ == '__main__':
    main()
