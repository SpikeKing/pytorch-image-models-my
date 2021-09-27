#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 27.9.21
"""

import os
import sys

from root_dir import DATA_DIR

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.make_html_page import make_html_page
from myutils.project_utils import *
from myutils.cv_utils import *


class DataProcessor(object):
    """
    数据处理类
    """
    def __init__(self):
        self.file_name = os.path.join(DATA_DIR, "files_v2", "文字检测_识别文字检测数据清洗_.txt")
        self.html_file_name = os.path.join(DATA_DIR, "files_v2", "文字检测_识别文字检测数据清洗_.html")

    @staticmethod
    def save_img_path(img_bgr, img_name, oss_root_dir=""):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        if not oss_root_dir:
            oss_root_dir = "zhengsheng.wcl/Doc-Clz/datasets/v2/{}".format(get_current_day_str())

        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    def process(self):
        print('[Info] 文件: {}'.format(self.file_name))
        data_lines = read_file(self.file_name)
        print('[Info] 文本行数: {}'.format(len(data_lines)))
        data_count_dict = collections.defaultdict(list)
        for data_line in data_lines:
            data_dict = json.loads(data_line)
            url = data_dict["url"]
            label = data_dict["cate"]
            data_count_dict[label].append(url)
        # print('[Info] data_count_dict: {}'.format(data_count_dict))

        random.seed(47)
        # 验证模型
        item_list = []
        for label in data_count_dict.keys():
            url_list = data_count_dict[label]
            random.shuffle(url_list)
            url_list = url_list[:100]
            for url in url_list:
                item_list.append([url, label])

        make_html_page(self.html_file_name, item_list)


def main():
    dp = DataProcessor()
    dp.process()


if __name__ == '__main__':
    main()
