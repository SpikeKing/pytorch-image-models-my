#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 24.9.21
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


class TextLineLabeledChecker(object):
    def __init__(self):
        # self.file1_path = os.path.join(DATA_DIR, "labeled", "labeled_5w_166537.1.csv")
        # self.out_file1_path = os.path.join(DATA_DIR, "labeled", "labeled_5w_166537.1-out.txt")
        self.file2_path = os.path.join(DATA_DIR, "labeled", "labeled_5w_166537.2.csv")
        self.out_file2_path = os.path.join(DATA_DIR, "labeled", "labeled_5w_166537.2-out.txt")
        self.level1_path = os.path.join(DATA_DIR, "labeled", "labeled_5w_166537.level1.txt")
        self.level1_html = os.path.join(DATA_DIR, "labeled", "labeled_5w_166537.level1.html")
        self.level2_path = os.path.join(DATA_DIR, "labeled", "labeled_5w_166537.level2.txt")
        self.level2_html = os.path.join(DATA_DIR, "labeled", "labeled_5w_166537.level2.html")

    @staticmethod
    def process_file(file_path, out_path):
        print('[Info] 处理文件: {}'.format(file_path))
        column_names, row_list = read_csv_file(file_path)
        print('[Info] 样本数: {}'.format(len(row_list)))
        print('[Info] column_names: {}'.format(column_names))
        radio_2_dict = {"0": "印刷公式", "1": "印刷文本", "2": "手写公式", "3": "手写文本", "4": "艺术字", "5": "其他"}
        out_list = []
        right_count = 0  # 正确
        for row_idx, row in enumerate(row_list):
            img_url, pre_label = json.loads(row["问题内容"])
            radio_1 = json.loads(row["回答内容"])["radio_1"]
            radio_2 = json.loads(row["回答内容"])["radio_2"]
            if radio_1 == "0":
                art_label = pre_label
            else:
                if not radio_2:
                    continue
                art_label = radio_2_dict[radio_2]
            if pre_label == art_label:
                right_count += 1
            out_list.append("{}\t{}\t{}".format(img_url, pre_label, art_label))

        print('[Info] 预标注 准确率: {}'.format(safe_div(right_count, len(row_list))))

        create_file(out_path)  # 创建file
        write_list_to_file(out_path, out_list)

    @staticmethod
    def check_labeled_file(file_path, html_path):
        print('[Info] level_path: {}'.format(file_path))
        data1_lines = read_file(file_path)
        random.seed(47)
        random.shuffle(data1_lines)
        data1_lines = data1_lines[:100]
        print('[Info] 样本数: {}'.format(len(data1_lines)))
        out_list = []
        for data1_line in data1_lines:
            items = data1_line.split("\t")
            out_list.append(items)
        make_html_page(html_path, out_list)
        print('[Info] 写入完成: {}'.format(html_path))

    def valuate_label_file(self):
        self.process_file(self.file2_path, self.out_file2_path)
        print('[Info] 处理文件: {}'.format(self.out_file2_path))
        data_lines = read_file(self.out_file2_path)
        print('[Info] 样本数: {}'.format(len(data_lines)))

        level1_dict = collections.defaultdict(list)
        level2_dict = collections.defaultdict(list)
        for data_line in data_lines:
            img_url, pre_label, art_label = data_line.split("\t")
            level1_dict[img_url].append(pre_label)
            level1_dict[img_url].append(art_label)
            level2_dict[img_url].append(art_label)

        level1_out_list = []
        for img_url in level1_dict.keys():
            label_list = level1_dict[img_url]
            label_list = list(set(label_list))
            if len(label_list) == 1:
                level1_out_list.append("{}\t{}".format(img_url, label_list[0]))
        print('[Info] level1样本数: {}'.format(len(level1_out_list)))
        create_file(self.level1_path)
        write_list_to_file(self.level1_path, level1_out_list)

        level2_out_list = []
        for img_url in level2_dict.keys():
            label_list = level2_dict[img_url]
            label_list = list(set(label_list))
            if len(label_list) == 1:
                level2_out_list.append("{}\t{}".format(img_url, label_list[0]))
        print('[Info] level2样本数: {}'.format(len(level2_out_list)))
        create_file(self.level2_path)
        write_list_to_file(self.level2_path, level2_out_list)

        create_file(self.level1_html)
        TextLineLabeledChecker.check_labeled_file(self.level1_path, self.level1_html)
        create_file(self.level2_html)
        TextLineLabeledChecker.check_labeled_file(self.level2_path, self.level2_html)

    @staticmethod
    def check_big_or_small(data_idx, data_line, small_file, big_file):
        url, label = data_line.split("\t")
        _, img_bgr = download_url_img(url)
        h, w, _ = img_bgr.shape
        # print('[Info] shape: {}'.format(img_bgr.shape))
        area = h * w / 25
        # print('[Info] area: {}'.format(area))
        if area < 3000:
            write_line(small_file, data_line)
        else:
            write_line(big_file, data_line)
        if data_idx % 1000 == 0:
            print('[Info] data_idx: {}'.format(data_idx))

    def split_big_and_small(self):
        print('[Info] 处理文件: {}'.format(self.level1_path))
        data_lines = read_file(self.level1_path)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        small_file = ".".join(self.level1_path.split(".")[:-1]) + ".small.txt"
        big_file = ".".join(self.level1_path.split(".")[:-1]) + ".big.txt"
        print('[Info] small_file: {}'.format(small_file))
        print('[Info] big_file: {}'.format(big_file))
        pool = Pool(processes=40)
        for data_idx, data_line in enumerate(data_lines):
            # TextLineLabeledChecker.check_big_or_small(data_idx, data_line, small_file, big_file)
            pool.apply_async(TextLineLabeledChecker.check_big_or_small, (data_idx, data_line, small_file, big_file))
        pool.close()
        pool.join()
        print('[Info] 全部处理完成!')


def main():
    tlc = TextLineLabeledChecker()
    # tlc.valuate_label_file()
    tlc.split_big_and_small()


if __name__ == '__main__':
    main()
