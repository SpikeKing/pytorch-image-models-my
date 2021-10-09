#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 9.10.21
"""

import os
import cv2
import time

from multiprocessing.pool import Pool

from myutils.project_utils import mkdir_if_not_exist, download_url_img


def process_sample_url(data_line, label_dir, label_idx, data_idx):
    """
    处理单个样本的url, 用于多进程
    """
    _, img_bgr = download_url_img(data_line)
    file_path = os.path.join(label_dir, "{}_{}.jpg".format(str(label_idx).zfill(3), str(data_idx).zfill(7)))
    cv2.imwrite(file_path, img_bgr)
    if data_idx % 1000 == 0:
        print("[Info] \t 已处理: {}".format(data_idx))
        time.sleep(10)  # 避免访问过快


def generate_dataset_mul(dataset_dir, train_list, val_list):
    """
    生成数据集
    @param dataset_dir 数据集地址
    @param train_list 训练集列表，每项都是一个类别的url列表
    @param val_list 验证集列表，每项都是一个类别的url列表
    """
    print('[Info] 数据集处理开始: {}'.format(dataset_dir))
    mkdir_if_not_exist(dataset_dir)
    train_dir = os.path.join(dataset_dir, "train")
    mkdir_if_not_exist(train_dir)
    val_dir = os.path.join(dataset_dir, "val")
    mkdir_if_not_exist(val_dir)

    def process_data_list(data_list_, pool_):
        for label_idx, data_lines in enumerate(data_list_):
            label_dir = os.path.join(train_dir, "{}".format(str(label_idx).zfill(3)))
            mkdir_if_not_exist(label_dir)
            for data_idx, data_line in enumerate(data_lines):
                pool_.apply_async(process_sample_url, (data_line, label_dir, label_idx, data_idx))

    pool = Pool(processes=100)
    process_data_list(train_list, pool)
    process_data_list(val_list, pool)
    pool.close()
    pool.join()
    print('[Info] 数据集处理完成: {}'.format(dataset_dir))

