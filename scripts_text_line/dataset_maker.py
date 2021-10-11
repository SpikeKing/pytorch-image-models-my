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
        self.in_folder = os.path.join(DATA_DIR, "datasets", "text_line_dataset_c4_20211011_files")
        self.out_folder = os.path.join(DATA_DIR, "datasets", "text_line_dataset_c4_20211011")

    def process(self):
        print('[Info] 输入文件夹: {}'.format(self.in_folder))
        paths_list, _ = traverse_dir_files(self.in_folder)
        new_paths_list = []
        for path in paths_list:
            tag = path.split("_")[-1]
            if tag == "rgt.txt":
                new_paths_list.append(path)
        print('[Info] 过滤文件数: {}'.format(len(new_paths_list)))

        train_list, val_list = [], []
        for path in new_paths_list:
            folder_name = path.split("/")[-2]
            type_name, label_name = folder_name.split("_")
            print('[Info] type_name: {}, label_name: {}'.format(type_name, label_name))
            data_lines = read_file(path)
            urls = []
            for data_line in data_lines:
                items = data_line.split("\t")
                urls.append(items[0])
            print('[Info] 样本数: {}'.format(urls))
            if type_name == "train":
                train_list.append(urls)
            else:
                val_list.append(urls)

        generate_dataset_mul(self.out_folder, train_list, val_list)


def main():
    dm = DatasetMaker()
    dm.process()


if __name__ == '__main__':
    main()
