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

from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import DATA_DIR


class TextLineLabeledDataset(object):
    def __init__(self):
        self.file1_path = os.path.join(DATA_DIR, "labeled", "labeled_5w_166537.level2.small.txt")
        self.file2_path = os.path.join(DATA_DIR, "labeled", "labeled_5w_166537.level2.big.txt")
        self.data_dict = {"其他": "0", "印刷公式": "1", "印刷文本": "2", "手写公式": "3", "手写文本": "4", "艺术字": "5"}

    @staticmethod
    def process_line(image_idx, img_url, img_path):
        _, img_bgr = download_url_img(img_url)
        cv2.imwrite(img_path, img_bgr)
        if image_idx % 1000 == 0:
            print('[Info] data_idx: {}'.format(image_idx))

    def process(self):
        dataset_folder = os.path.join(DATA_DIR, "datasets")
        mkdir_if_not_exist(dataset_folder)
        # file_path = self.file1_path
        # dataset_path = os.path.join(dataset_folder, "text_line_v2_small")
        file_path = self.file2_path
        dataset_path = os.path.join(dataset_folder, "text_line_v2_big")
        mkdir_if_not_exist(dataset_path)

        print('[Info] 文件路径: {}'.format(file_path))
        data_lines = read_file(file_path)
        random.seed(47)
        random.shuffle(data_lines)
        print("[Info] 样本数: {}".format(len(data_lines)))
        n = 15
        gap = len(data_lines) // n
        train_lines = data_lines[:gap*(n-1)]
        val_lines = data_lines[gap*(n-1):]
        print("[Info] 训练: {}, 测试: {}".format(len(train_lines), len(val_lines)))

        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(train_lines):
            img_url, label_str = data_line.split("\t")
            ds_type = "train"
            train_path = os.path.join(dataset_path, ds_type)
            label = self.data_dict[label_str].zfill(3)
            label_dir = os.path.join(train_path, label)
            mkdir_if_not_exist(label_dir)
            img_path = os.path.join(label_dir, "{}_{}_{}.jpg".format(ds_type, label, str(data_idx).zfill(7)))
            pool.apply_async(TextLineLabeledDataset.process_line, (data_idx, img_url, img_path))
            # TextLineLabeledDataset.process_line(data_idx, img_url, img_path)
            # if data_idx == 10:
            #     break

        for data_idx, data_line in enumerate(val_lines):
            img_url, label_str = data_line.split("\t")
            ds_type = "val"
            val_path = os.path.join(dataset_path, ds_type)
            label = self.data_dict[label_str].zfill(3)
            label_dir = os.path.join(val_path, label)
            mkdir_if_not_exist(label_dir)
            img_path = os.path.join(label_dir, "{}_{}_{}.jpg".format(ds_type, label, str(data_idx).zfill(7)))
            pool.apply_async(TextLineLabeledDataset.process_line, (data_idx, img_url, img_path))
            # TextLineLabeledDataset.process_line(data_idx, img_url, img_path)
            # if data_idx == 10:
            #     break

        pool.close()
        pool.join()


def main():
    tld = TextLineLabeledDataset()
    tld.process()


if __name__ == '__main__':
    main()
