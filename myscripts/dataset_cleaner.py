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

from myutils.make_html_page import make_html_page
from x_utils.vpf_sevices import get_vpf_service_np
from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import DATA_DIR


class DatasetCleaner(object):
    """
    数据集清理
    """
    def __init__(self):
        self.dataset_folder = os.path.join(DATA_DIR, "datasets", "document_dataset")
        self.dataset_folder_path = os.path.join(DATA_DIR, "files_v2", "document_dataset_path.txt")
        self.error_path = os.path.join(DATA_DIR, "files_v2", "document_dataset_error.{}.txt".format(get_current_time_str()))
        self.error_html_path = os.path.join(DATA_DIR, "files_v2", "document_dataset_error.html")

    @staticmethod
    def call_service(img_bgr):
        """
        调用服务
        """
        data_dict = get_vpf_service_np(img_bgr, service_name="7VKS8amEShUQmmFWrhKWee")
        label = data_dict["data"]["label"]
        prob_list = data_dict["data"]["prob_list"]
        return label, prob_list

    @staticmethod
    def save_img_path(img_bgr, img_name, oss_root_dir=""):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        if not oss_root_dir:
            oss_root_dir = "zhengsheng.wcl/Doc-Clz/datasets/error/{}".format(get_current_day_str())
        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    @staticmethod
    def process_line(data_idx, data_line, error_path):
        img_bgr = cv2.imread(data_line)
        label_str, prob_list = DatasetCleaner.call_service(img_bgr)
        pre_label_str = data_line.split("/")[-2]
        label = int(label_str)
        pre_label = int(pre_label_str)
        if label == pre_label:
            return
        else:
            label_dict = {0: "纸质文档", 1: "非纸质文档"}
            pre_label = label_dict[pre_label]
            label = label_dict[label]
            img_name = "{}-{}.jpg".format(data_idx, get_current_time_str())
            img_url = DatasetCleaner.save_img_path(img_bgr, img_name)
            write_line(error_path, "{}\t{}\t{}\t{}".format(img_url, pre_label, label, data_line))
        print('[Info] 处理完成: {}'.format(data_idx))

    def process(self):
        print('[Info] 读取文件: {}'.format(self.dataset_folder_path))
        data_lines = read_file(self.dataset_folder_path)
        n = len(data_lines)
        if n == 0:  # 加载文件
            print('[Info] 读取文件夹: {}'.format(self.dataset_folder))
            paths_list, names_list = traverse_dir_files(self.dataset_folder)
            write_list_to_file(self.dataset_folder_path, paths_list)
            data_lines = paths_list
        else:
            print('[Info] 使用已有文件: {}'.format(self.dataset_folder_path))
        print("[Info] 样本数: {}".format(len(data_lines)))

        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(data_lines):
            # DatasetCleaner.process_line(data_idx, data_line, self.error_path)
            pool.apply_async(DatasetCleaner.process_line, (data_idx, data_line, self.error_path))
        pool.close()
        pool.join()

        error_lines = read_file(self.error_path)
        print('[Info] 正确率: {}'.format(safe_div(len(error_lines), len(data_lines))))
        items = []
        for line in error_lines:
            img_url, pre_label, label, data_line = line.split("\t")
            items.append([img_url, pre_label, label, data_line])
        create_file(self.error_html_path)
        make_html_page(self.error_html_path, items)
        print('[Info] 写入完成: {}'.format(self.error_html_path))


def main():
    dc = DatasetCleaner()
    dc.process()


if __name__ == '__main__':
    main()
