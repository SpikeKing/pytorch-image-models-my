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
    def load_data_file(file_list, discard=4, key="radio_1"):
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

        data_dict = collections.defaultdict(list)
        for row_idx, row in enumerate(row_list):
            url = eval(row["问题内容"])[0]
            label = eval(row["回答内容"])[key]
            if not label:
                continue
            data_dict[url].append(label)

        right_dict = collections.defaultdict(list)
        for url in data_dict.keys():
            raw_labels = data_dict[url]
            labels = list(set(raw_labels))
            if len(labels) == 1 and len(raw_labels) == 2:
                label = labels[0]
                if label and int(label) == discard:
                    continue
                right_dict[label].append(url)

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
    def save_data(right_dict, label_dict, label_format_name, html_name, discard=4):
        print("-" * 200)
        all_num = 0
        for label in right_dict.keys():
            if int(label) == discard:
                continue
            num = len(right_dict[label])
            all_num += num
        print('[Info] 样本总数: {}'.format(all_num))

        pool = Pool(processes=100)
        for label in right_dict.keys():
            print('[Info] ' + "-" * 50)
            print('[Info] label: {}'.format(label_dict[int(label)]))
            print('[Info] 样本数: {}'.format(len(right_dict[label])))
            print('[Info] 占比: {} %'.format(safe_div(len(right_dict[label]), all_num) * 100))
            label_file = label_format_name.format(label_dict[int(label)])
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
            label_str = label_dict[int(label)]
            for url in urls:
                items.append([url, label_str])

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

        right_dict = Labeled2Dataset.load_data_file([file1_name, file2_name, file3_name], discard=-1, key="radio_2")

        Labeled2Dataset.save_data(right_dict, label_dict, label_format_name, html_name, discard=-1)

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
        right1_dict = Labeled2Dataset.load_data_file([file1_name, file2_name])
        right2_dict = Labeled2Dataset.load_data_file([file3_name])

        right_dict = Labeled2Dataset.merge_data_dict(right1_dict, right2_dict)

        Labeled2Dataset.save_data(right_dict, label_dict, label_format_name, html_name)


def main():
    l2d = Labeled2Dataset()
    l2d.process_v1()
    # l2d.filter_labeled_file()


if __name__ == '__main__':
    main()
