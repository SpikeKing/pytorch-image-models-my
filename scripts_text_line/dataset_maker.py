#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 11.10.21
"""

import os
import sys


p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myscripts.dataset_utils import generate_dataset_mul
from myutils.project_utils import *
from root_dir import DATA_DIR


class DatasetMaker(object):
    def __init__(self):
        self.in_folder = os.path.join(DATA_DIR, "datasets", "text_line_dataset_c4_20211013_files")
        # self.out_folder = os.path.join(DATA_DIR, "datasets", "text_line_dataset_c4_20211013")
        self.out_folder = os.path.join(DATA_DIR, "datasets", "text_line_dataset_c4_20211013_square")

    def process(self):
        print('[Info] 输入文件夹: {}'.format(self.in_folder))
        paths_list, _ = traverse_dir_files(self.in_folder)

        train_list, val_list = [], []
        for path in paths_list:
            folder_name = path.split("/")[-1]
            type_name, label_name = folder_name.split("_")
            print('[Info] type_name: {}, label_name: {}'.format(type_name, label_name))
            data_lines = read_file(path)
            urls = []
            for data_line in data_lines:
                items = data_line.split("\t")
                urls.append(items[0])
            print('[Info] 样本数: {}'.format(len(urls)))
            if type_name == "train":
                train_list.append(urls)
            else:
                val_list.append(urls)

        generate_dataset_mul(self.out_folder, train_list, val_list, is_square=True)


def main():
    dm = DatasetMaker()
    dm.process()


if __name__ == '__main__':
    main()
