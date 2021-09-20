#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 17.9.21
"""

import os
import sys

from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import DATA_DIR, ROOT_DIR


class TextLineAnalyzer(object):
    def __init__(self):
        self.file_name = os.path.join(DATA_DIR, "files", "common_full_out_v1_200w.txt")
        self.out_folder = os.path.join(DATA_DIR, "files", "common_full_out_v1_200w")
        self.ds_folder = os.path.join(ROOT_DIR, "..", "datasets", "text_line_v1_1_200w")
        self.ds_folder_txt = os.path.join(ROOT_DIR, "..", "datasets", "text_line_v1_1_200w.txt")  # 存储路径文件
        mkdir_if_not_exist(self.out_folder)
        mkdir_if_not_exist(self.ds_folder)

    def split_train_and_val(self):
        print('[Info] 待处理文件: {}'.format(self.file_name))
        data_lines = read_file(self.file_name)
        print('[Info] 样本数: {}'.format(len(data_lines)))

        data_num_dict = collections.defaultdict(int)
        data_urls_dict = collections.defaultdict(list)
        for data_line in data_lines:
            img_url, label = data_line.split("\t")
            data_num_dict[label] += 1
            data_urls_dict[label].append(img_url)
        print('[Info] 样本分布: {}'.format(data_num_dict))

        ratio = 0.95
        for label in data_urls_dict.keys():
            url_list = data_urls_dict[label]
            b = int(len(url_list) * ratio)
            random.shuffle(url_list)
            train_urls = url_list[:b]
            val_urls = url_list[b:]
            train_file = os.path.join(self.out_folder, "{}_train.txt".format(str(label).zfill(3)))
            val_file = os.path.join(self.out_folder, "{}_val.txt".format(str(label).zfill(3)))
            write_list_to_file(train_file, train_urls)
            write_list_to_file(val_file, val_urls)
        print('[Info] 数据集拆分完成: {}'.format(self.out_folder))

    @staticmethod
    def process_line(img_idx, img_url, img_folder, img_folder_txt, ds_type):
        _, img_bgr = download_url_img(img_url)
        img_path = os.path.join(img_folder, "{}_{}.jpg".format(ds_type, str(img_idx).zfill(7)))
        cv2.imwrite(img_path, img_bgr)
        write_line(img_folder_txt, img_path)
        if img_idx % 1000 == 0:
            print('[Info] img_idx: {}'.format(img_idx))

    @staticmethod
    def process_line_try(img_idx, img_url, img_folder, img_folder_txt, ds_type):
        try:
            TextLineAnalyzer.process_line(img_idx, img_url, img_folder, img_folder_txt, ds_type)
        except Exception as e:
            print('[Error] {}, {}'.format(img_idx, img_url))
            print('[Error] {}'.format(e))

    def create_dataset(self, is_balance):
        """
        创建DIR
        """
        print('[Info] 数据文件夹: {}'.format(self.out_folder))
        paths_list, names_list = traverse_dir_files(self.out_folder)
        print('[Info] 数据集文件夹: {}'.format(self.ds_folder))
        print('[Info] 数据集文件夹: {}'.format(self.ds_folder_txt))
        pool = Pool(processes=100)
        for path in paths_list:
            file_name = path.split("/")[-1].split(".")[0]
            label, ds_type = file_name.split("_")

            # 创建dir
            ds_type_folder = os.path.join(self.ds_folder, ds_type)
            mkdir_if_not_exist(ds_type_folder)
            label_dir = os.path.join(ds_type_folder, label)
            mkdir_if_not_exist(label_dir)

            data_lines = read_file(path)
            if ds_type == "train":
                sample_num = 500000
                # sample_num = 5
            elif ds_type == "val":
                sample_num = 30000
                # sample_num = 2
            else:
                raise Exception("ds_type error")

            if is_balance:
                url_list = get_fixed_samples(data_lines, sample_num)
            else:
                url_list = data_lines
            for img_idx, img_url in enumerate(url_list):
                # TextLineAnalyzer.process_line_try(img_idx, img_url, label_dir, ds_type)
                pool.apply_async(TextLineAnalyzer.process_line_try,
                                 (img_idx, img_url, label_dir, self.ds_folder_txt, ds_type))

        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(self.ds_folder))


def main():
    tla = TextLineAnalyzer()
    tla.create_dataset(is_balance=False)


if __name__ == '__main__':
    main()
