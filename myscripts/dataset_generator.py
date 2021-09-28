#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 28.9.21
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


class DatasetGenerator(object):
    def __init__(self):
        self.dataset_path = os.path.join(DATA_DIR, "datasets", "ch_en_line_v1")
        self.train_file = os.path.join(DATA_DIR, 'files_ch_en', "line_cls_train.txt")
        self.val_file = os.path.join(DATA_DIR, 'files_ch_en', "line_cls_train.txt")
        self.max_num = 100000
        # self.max_num = 20

    @staticmethod
    def process_line(img_idx, img_url, img_label, data_type, folder_path):
        _, img_bgr = download_url_img(img_url)
        img_path = os.path.join(folder_path, "{}_{}_{}.jpg".format(data_type, str(img_idx).zfill(8), img_label))
        cv2.imwrite(img_path, img_bgr)
        if img_idx % 1000 == 0:
            print('[Info] img_idx: {}'.format(img_idx))

    @staticmethod
    def process_file(file_path, max_num, ds_folder, data_type):
        """
        处理单个文件
        """
        print('[Info] 文件路径: {}'.format(file_path))
        print('[Info] max_num: {}'.format(max_num))
        print('[Info] ds_folder: {}'.format(ds_folder))
        print('[Info] data_type: {}'.format(data_type))

        mkdir_if_not_exist(ds_folder)
        type_folder = os.path.join(ds_folder, data_type)
        mkdir_if_not_exist(type_folder)
        print('[Info] 类型数据集: {}'.format(type_folder))

        data_lines = read_file(file_path)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        n = len(data_lines)
        if n == 0:
            return
        random.seed(47)
        random.shuffle(data_lines)
        n = len(data_lines)
        if n > max_num:
            data_lines = data_lines[:max_num]
        print('[Info] 样本数: {}'.format(len(data_lines)))

        sym = ""
        x1 = data_lines[0].split("\t")
        x2 = data_lines[0].split(" ")
        if len(x1) >= 2:
            sym = "\t"
        elif len(x2) >= 2:
            sym = " "
        if not sym:
            return

        process_list = []
        for img_idx, data_line in enumerate(data_lines):
            items = data_line.split(sym)
            img_url = items[0]
            img_label = items[1]
            img_label = str(img_label).zfill(3)
            label_folder = os.path.join(type_folder, img_label)
            mkdir_if_not_exist(label_folder)
            process_list.append([img_idx, img_url, img_label, data_type, label_folder])
        return process_list

    def process(self):
        train_max = self.max_num
        val_max = self.max_num // 10
        train_list = DatasetGenerator.process_file(self.train_file, train_max, self.dataset_path, "train")
        val_list = DatasetGenerator.process_file(self.val_file, val_max, self.dataset_path, "val")
        print('[Info] 训练样本: {}, 验证样本: {}'.format(len(train_list), len(val_list)))
        process_list = train_list + val_list

        pool = Pool(processes=100)
        for items in process_list:
            img_idx, img_url, img_label, data_type, label_folder = items
            # DatasetGenerator.process_line(img_idx, img_url, img_label, data_type, label_folder)
            pool.apply_async(DatasetGenerator.process_line, (img_idx, img_url, img_label, data_type, label_folder))
        pool.close()
        pool.join()
        print('[Info] 数据集处理完成: {}'.format(self.dataset_path))


def main():
    dg = DatasetGenerator()
    dg.process()


if __name__ == '__main__':
    main()
