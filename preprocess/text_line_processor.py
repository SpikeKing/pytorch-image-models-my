#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 16.9.21
"""

import os
import sys

from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from myutils.cv_utils import *
from myutils.crop_img_from_coord import crop_img_from_coord
from root_dir import DATA_DIR


class TextLineProcessor(object):
    """
    文本行分类数据集的创建类
    """
    def __init__(self):
        self.file_path = os.path.join(DATA_DIR, "files", "common_full.txt")  # 文件
        self.out_file_path = os.path.join(DATA_DIR, "files",
                                          "common_full_out.{}.txt".format(get_current_time_str()))  # 输出文件

    @staticmethod
    def save_img_path(img_bgr, img_name, oss_root_dir=""):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        if not oss_root_dir:
            oss_root_dir = "zhengsheng.wcl/Text-Line-Clz/datasets/v1/{}".format(get_current_day_str())
        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    @staticmethod
    def process_line(data_idx, data_line, out_file_path):
        data = eval(data_line.strip())
        url = data["url"]
        labels = data['label']
        coords = data['coord']
        img_name_x = url.split("/")[-1].split(".")[0]  # 图像名称
        _, img_bgr = download_url_img(url)
        for i in range(len(labels)):
            label = labels[i]
            if label in [1, 2, 3, 4]:
                coord = np.array(coords[i])
                crop_img = crop_img_from_coord(coord, img_bgr)
                img_name = "{}_s_{}_s_{}.jpg".format(img_name_x, str(i), str(label))
                img_out = resize_crop_square(crop_img)
                img_url = TextLineProcessor.save_img_path(img_out, img_name)
                write_line(out_file_path, "{}\t{}".format(img_url, str(label)))
        if data_idx % 1000 == 0:
            print('[Info] 处理完成: {}'.format(data_idx))

    def process(self):
        print('[Info] 处理文件: {}'.format(self.file_path))
        data_lines = read_file(self.file_path)
        print('[Info] 样本数: {}'.format(len(data_lines)))

        pool = Pool(processes=40)
        for data_idx, data_line in enumerate(data_lines):
            pool.apply_async(TextLineProcessor.process_line, (data_idx, data_line, self.out_file_path))
            # TextLineProcessor.process_line(data_idx, data_line, self.out_file_path)
            # break
        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(self.out_file_path))


def main():
    tlp = TextLineProcessor()
    tlp.process()


if __name__ == "__main__":
    main()
