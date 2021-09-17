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
    def get_negative_box(img_bgr, coords):
        """
        获取负例的box
        """
        h, w, _ = img_bgr.shape
        n = len(coords)
        bbox_list = []
        for i in range(n):
            bbox = rec2bbox(coords[i])
            bbox_list.append(bbox)

        # 用于计算iou
        img_mask = np.zeros((h, w))
        img_mask = img_mask.astype(np.uint8)
        avg_x_list, avg_y_list = [], []
        for box in bbox_list:
            x_min, y_min, x_max, y_max = box
            img_mask[y_min:y_max, x_min:x_max] = 1
            if min(x_max - x_min, y_max - y_min) < 30:
                continue
            avg_x_list.append(x_max - x_min)
            avg_y_list.append(y_max - y_min)

        new_n = len(avg_x_list)
        avg_x = int(np.average(avg_x_list))
        avg_y = int(np.average(avg_y_list))

        # 随机剪裁
        x_step = avg_x
        y_step = avg_y
        num_w = w // avg_x
        num_h = h // avg_y

        # 随机生成
        box_list = []
        for i in range(num_w):
            for j in range(num_h):
                s_x = max(i*x_step, 0)
                e_x = min(s_x + x_step, w)
                s_y = max(j*y_step, 0)
                e_y = min(s_y + y_step, h)
                iou = np.sum(img_mask[s_y:e_y, s_x:e_x]) / ((e_x - s_x) * (e_y - s_y))
                if iou < 0.1:
                    bbox = [s_x, s_y, e_x, e_y]
                    box_list.append(bbox)

        # 随机采样
        random.shuffle(box_list)
        box_list = box_list[:new_n // 4]
        return box_list


    @staticmethod
    def process_line(data_idx, data_line, out_file_path):
        data = eval(data_line.strip())
        url = data["url"]
        labels = data['label']
        coords = data['coord']
        img_name_x = url.split("/")[-1].split(".")[0]  # 图像名称
        _, img_bgr = download_url_img(url)

        # 处理标签
        for i in range(len(labels)):
            label = labels[i]
            if label in [1, 2, 3, 4]:
                coord = np.array(coords[i])
                crop_img = crop_img_from_coord(coord, img_bgr)
                img_name = "{}_s_{}_s_{}.jpg".format(img_name_x, str(i), str(label))
                img_out = resize_crop_square(crop_img)
                h, w, _ = img_out.shape
                if min(h, w) < 30:
                    continue
                img_url = TextLineProcessor.save_img_path(img_out, img_name)
                write_line(out_file_path, "{}\t{}".format(img_url, str(label)))

        # 处理负例
        neg_boxes = TextLineProcessor.get_negative_box(img_bgr, coords)
        for i, neg_box in enumerate(neg_boxes):
            crop_img = get_cropped_patch(img_bgr, neg_box)
            img_out = resize_crop_square(crop_img)
            neg_label = 0
            # 负例标签是0
            img_name = "{}_s_{}_s_{}.jpg".format(img_name_x, str(i + len(labels)), str(neg_label))
            img_url = TextLineProcessor.save_img_path(img_out, img_name)
            write_line(out_file_path, "{}\t{}".format(img_url, str(neg_label)))

        if data_idx % 1000 == 0:
            print('[Info] 处理完成: {}'.format(data_idx))

    @staticmethod
    def process_line_try(data_idx, data_line, out_file_path):
        try:
            TextLineProcessor.process_line(data_idx, data_line, out_file_path)
        except Exception as e:
            print('[Error] data_idx: {}'.format(data_idx))
            print('[Error] e: {}'.format(e))

    def process(self):
        print('[Info] 处理文件: {}'.format(self.file_path))
        data_lines = read_file(self.file_path)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        # random.seed(47)
        # random.shuffle(data_lines)

        pool = Pool(processes=40)
        for data_idx, data_line in enumerate(data_lines):
            pool.apply_async(TextLineProcessor.process_line_try, (data_idx, data_line, self.out_file_path))
            # TextLineProcessor.process_line_try(data_idx, data_line, self.out_file_path)
            # break
        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(self.out_file_path))


def main():
    tlp = TextLineProcessor()
    tlp.process()


if __name__ == "__main__":
    main()
