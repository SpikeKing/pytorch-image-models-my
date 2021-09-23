#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 23.9.21
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


class TextLineLabeled(object):
    """
    文本行分类数据集的创建类
    """
    def __init__(self):
        folder_path = os.path.join(DATA_DIR, "files")
        self.file_path = os.path.join(folder_path, "4wedu+2wopensource+2.5wnature.labeled-10000.1.txt")
        self.out_file_path = \
            os.path.join(folder_path, "4wedu+2wopensource+2.5wnature.labeled-10000.1.out{}.txt"
                         .format(get_current_time_str()))

    @staticmethod
    def save_img_path(img_bgr, img_name, oss_root_dir=""):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        if not oss_root_dir:
            oss_root_dir = "zhengsheng.wcl/Text-Line-Clz/datasets/v1_2/{}".format(get_current_day_str())

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

        if not avg_x_list:
            return []

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
                if iou < 0.001:
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
            if label in [1, 2, 3, 4, 5]:
                coord = np.array(coords[i])
                crop_img = crop_img_from_coord(coord, img_bgr)
                crop_img = cv2.resize(crop_img, None, fx=5.0, fy=5.0)   # 有助于提升标注效率
                img_name = "{}_s_{}_s_{}.jpg".format(img_name_x, str(i), str(label))
                img_out = crop_img
                h, w, _ = img_out.shape
                if h * w < 2000:
                    continue
                img_url = TextLineLabeled.save_img_path(img_out, img_name)
                write_line(out_file_path, "{}\t{}".format(img_url, str(label)))

        # 处理负例
        neg_boxes = TextLineLabeled.get_negative_box(img_bgr, coords)
        for i, neg_box in enumerate(neg_boxes):
            neg_label = 0  # 负例标签是0
            crop_img = get_cropped_patch(img_bgr, neg_box)
            img_name = "{}_s_{}_s_{}.jpg".format(img_name_x, str(i + len(labels)), str(neg_label))
            img_out = crop_img

            img_url = TextLineLabeled.save_img_path(img_out, img_name)
            write_line(out_file_path, "{}\t{}".format(img_url, str(neg_label)))

        if data_idx % 1000 == 0:
            print('[Info] 处理完成: {}'.format(data_idx))

    @staticmethod
    def process_line_try(data_idx, data_line, out_file_path):
        try:
            TextLineLabeled.process_line(data_idx, data_line, out_file_path)
        except Exception as e:
            print('[Error] data_idx: {}'.format(data_idx))
            print('[Error] e: {}'.format(e))

    def process(self):
        print('[Info] 处理文件: {}'.format(self.file_path))
        data_lines = read_file(self.file_path)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        random.seed(47)
        random.shuffle(data_lines)

        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(data_lines):
            # TextLineLabeled.process_line_try(data_idx, data_line, self.out_file_path)
            # if data_idx == 10:
            #     break
            pool.apply_async(TextLineLabeled.process_line_try, (data_idx, data_line, self.out_file_path))
        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(self.out_file_path))

    @staticmethod
    def split_labeled_files():
        """
        生成待标注样本
        """
        file_path = os.path.join(DATA_DIR, "files", "4wedu+2wopensource+2.5wnature.txt")
        out_file1_format = os.path.join(DATA_DIR, "files", "4wedu+2wopensource+2.5wnature.labeled-{}.txt")
        out_file2_format = os.path.join(DATA_DIR, "files", "4wedu+2wopensource+2.5wnature.rest-{}.txt")
        print('[Info] 文件路径: {}'.format(file_path))
        data_lines = read_file(file_path)
        random.seed(47)  # 需要随机
        random.shuffle(data_lines)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        n = 10000
        labeled_lines = data_lines[:n]
        rest_lines = data_lines[n:]
        print('[Info] 标注行: {}'.format(len(labeled_lines)))
        print('[Info] 剩余行: {}'.format(len(rest_lines)))
        write_list_to_file(out_file1_format.format(len(labeled_lines)), labeled_lines)
        write_list_to_file(out_file2_format.format(len(rest_lines)), rest_lines)
        print('[Info] 处理完成: {}'.format(file_path))

    @staticmethod
    def generate_labeled():
        file_path = os.path.join(DATA_DIR, "files", "4wedu+2wopensource+2.5wnature.labeled-10000.1.out.txt")
        out_file1_path = os.path.join(DATA_DIR, "files", "4wedu+2wopensource+2.5wnature.labeled-10000.1.out1.xlsx")
        out_file2_path = os.path.join(DATA_DIR, "files", "4wedu+2wopensource+2.5wnature.labeled-10000.1.out2.xlsx")
        print('[Info] 文件名: {}'.format(file_path))
        data_lines = read_file(file_path)
        print('[Info] 文本行数: {}'.format(len(data_lines)))

        item_list = []
        for data_line in data_lines:
            url, label = data_line.split("\t")
            label_str_dict = {"0": "其他", "1": "印刷公式", "2": "印刷文本", "3": "手写公式", "4": "手写文本", "5": "艺术字"}
            label_str = label_str_dict[label]
            item_list.append([url, label_str])

        write_list_to_excel(out_file1_path, ["url", "预标签"], item_list[:50000])
        write_list_to_excel(out_file2_path, ["url", "预标签"], item_list[50000:])


def main():
    tlp = TextLineLabeled()
    # tlp.split_labeled_files()
    tlp.process()
    # tlp.generate_labeled()


if __name__ == "__main__":
    main()
