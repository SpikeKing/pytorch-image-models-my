#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 23.9.21
"""

import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from root_dir import DATA_DIR


class LabelDataPrepare(object):
    """
    文本行分类数据集的创建类
    """
    def __init__(self):
        folder_path = os.path.join(DATA_DIR, "files_v2")
        self.all_file_path = os.path.join(folder_path, "text_line_nat_dataset.min100x100.txt")
        self.file1_path = os.path.join(folder_path, "text_line_nat_dataset.min100x100.0-50000.txt")
        self.file2_path = os.path.join(folder_path, "text_line_nat_dataset.min100x100.50000-75616.txt")

    def split_and_generate_files(self):
        """
        拆分原始样本
        """
        file_path = self.all_file_path  # 整体文档
        file_path_x = file_path.replace(".txt", "")

        label_str_dict = {"0": "其他", "1": "印刷公式", "2": "印刷文本", "3": "手写公式", "4": "手写文本", "5": "艺术字"}
        data_lines = read_file(file_path)
        n = len(data_lines)
        random.seed(47)  # 需要随机
        random.shuffle(data_lines)
        print('[Info] 样本数: {}'.format(n))
        gap = 50000
        for i in range(0, n, gap):
            s = i
            e = min(i + gap, n)
            sub_file_path = "{}.{}-{}.txt".format(file_path_x, s, e)
            sub_lines = data_lines[s:e]  # sub_lines
            print('[Info] txt: {}'.format(len(sub_lines)))
            create_file(sub_file_path)
            write_list_to_file(sub_file_path, sub_lines)

            sub_excel_path = "{}.{}-{}.xlsx".format(file_path_x, s, e)
            item_list = []
            for sub_line in sub_lines:
                data_dict = json.loads(sub_line)
                img_url = data_dict["crop_img_url"]
                # img_label = data_dict["label"]
                # label_str = label_str_dict[str(img_label)]
                item_list.append([img_url])
            print('[Info] excel: {}'.format(len(item_list)))
            create_file(sub_excel_path)
            write_list_to_excel(sub_excel_path, ["url"], item_list)

        print('[Info] 全部处理完成!')


def main():
    ldp = LabelDataPrepare()
    ldp.split_and_generate_files()


if __name__ == "__main__":
    main()
