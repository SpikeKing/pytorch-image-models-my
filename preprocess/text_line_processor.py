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

from myutils.make_html_page import make_html_page
from myutils.project_utils import *
from myutils.cv_utils import *
from myutils.crop_img_from_coord import crop_img_from_coord
from root_dir import DATA_DIR


class TextLineProcessor(object):
    """
    文本行分类数据集的创建类
    """
    def __init__(self):
        # v1
        # self.file_path = os.path.join(DATA_DIR, "files", "common_full.txt")  # 文件
        # self.out_file_path = os.path.join(DATA_DIR, "files",
        #                                   "common_full_out.{}.txt".format(get_current_time_str()))  # 输出文件
        # self.mini_file_path = os.path.join(DATA_DIR, "files", "common_full_mini.txt")  # 输出文件

        # v2
        self.file_path = os.path.join(DATA_DIR, "files", "4wedu+2wopensource+2.5wnature.labeled-10000.1.txt")
        self.train_file_path = os.path.join(DATA_DIR, "files", "text_line_dataset_v1_1_raw.train.txt")
        self.val_file_path = os.path.join(DATA_DIR, "files", "text_line_dataset_v1_1_raw.val.txt")

        # 输出文件
        self.out_train_file_path = \
            os.path.join(DATA_DIR, "files", "text_line_dataset_v1_1.train.{}.txt".format(get_current_time_str()))
        self.out_val_file_path = \
            os.path.join(DATA_DIR, "files", "text_line_dataset_v1_1.val.{}.txt".format(get_current_time_str()))

        self.mini_file_path = os.path.join(DATA_DIR, "files", "text_line_dataset_v1_1_mini.txt")  # 输出文件

    @staticmethod
    def save_img_path(img_bgr, img_name, oss_root_dir="", is_square=False):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        if not oss_root_dir:
            if is_square:
                oss_root_dir = "zhengsheng.wcl/Text-Line-Clz/datasets/v1_1_square/{}".format(get_current_day_str())
            else:
                oss_root_dir = "zhengsheng.wcl/Text-Line-Clz/datasets/v1_1/{}".format(get_current_day_str())

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
    def process_line(data_idx, data_line, out_file_path, is_square=True):
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
                if is_square:
                    img_name = "{}_s_{}_s_{}_square.jpg".format(img_name_x, str(i), str(label))
                    img_out = resize_crop_square(crop_img)
                else:
                    img_name = "{}_s_{}_s_{}.jpg".format(img_name_x, str(i), str(label))
                    img_out = crop_img
                h, w, _ = img_out.shape
                if min(h, w) < 20:
                    continue
                img_url = TextLineProcessor.save_img_path(img_out, img_name, is_square=is_square)
                write_line(out_file_path, "{}\t{}".format(img_url, str(label)))

        # 处理负例
        neg_boxes = TextLineProcessor.get_negative_box(img_bgr, coords)
        for i, neg_box in enumerate(neg_boxes):
            neg_label = 0  # 负例标签是0
            crop_img = get_cropped_patch(img_bgr, neg_box)
            if is_square:
                img_name = "{}_s_{}_s_{}_square.jpg".format(img_name_x, str(i + len(labels)), str(neg_label))
                img_out = resize_crop_square(crop_img)
            else:
                img_name = "{}_s_{}_s_{}.jpg".format(img_name_x, str(i + len(labels)), str(neg_label))
                img_out = crop_img

            img_url = TextLineProcessor.save_img_path(img_out, img_name, is_square=is_square)
            write_line(out_file_path, "{}\t{}".format(img_url, str(neg_label)))

        if data_idx % 1000 == 0:
            print('[Info] 处理完成: {}'.format(data_idx))

    @staticmethod
    def process_line_try(data_idx, data_line, out_file_path, is_square):
        try:
            TextLineProcessor.process_line(data_idx, data_line, out_file_path, is_square)
        except Exception as e:
            print('[Error] data_idx: {}'.format(data_idx))
            print('[Error] e: {}'.format(e))

    def split_train_and_val(self):
        """
        将样本拆分为训练数据和验证数据
        """
        print('[Info] 处理文件: {}'.format(self.file_path))
        data_lines = read_file(self.file_path)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        random.seed(47)
        random.shuffle(data_lines)
        n_split = len(data_lines) // 20 * 19
        train_data_lines = data_lines[:n_split]
        val_data_lines = data_lines[n_split:]
        print('[Info] 训练数据: {}, 验证数据: {}'.format(len(train_data_lines), len(val_data_lines)))
        if not os.path.exists(self.train_file_path):
            write_list_to_file(self.train_file_path, train_data_lines)
            write_list_to_file(self.val_file_path, val_data_lines)
            print('[Info] 写入完成: {}'.format(self.train_file_path))
            print('[Info] 写入完成: {}'.format(self.val_file_path))
        else:
            print('[Info] 已完成: {}'.format(self.train_file_path))
            print('[Info] 已完成: {}'.format(self.val_file_path))

    def process(self):
        print('[Info] 处理文件: {}'.format(self.train_file_path))
        print('[Info] 处理文件: {}'.format(self.val_file_path))
        train_data_lines = read_file(self.train_file_path)
        val_data_lines = read_file(self.val_file_path)
        print('[Info] 样本数: {}'.format(len(train_data_lines)))
        print('[Info] 样本数: {}'.format(len(val_data_lines)))
        random.seed(47)
        random.shuffle(train_data_lines)
        random.shuffle(val_data_lines)

        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(train_data_lines):
            # TextLineProcessor.process_line_try(data_idx, data_line, self.out_train_file_path, is_square=False)
            pool.apply_async(TextLineProcessor.process_line_try, (data_idx, data_line, self.out_train_file_path, False))
        for data_idx, data_line in enumerate(val_data_lines):
            # TextLineProcessor.process_line_try(data_idx, data_line, self.out_val_file_path)
            pool.apply_async(TextLineProcessor.process_line_try, (data_idx, data_line, self.out_val_file_path, False))
        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(self.out_train_file_path))
        print('[Info] 处理完成: {}'.format(self.out_val_file_path))

    @staticmethod
    def show_num_dict(num_dict):
        items = sort_dict_by_value(num_dict)
        all_num = 0
        for _, num in items:
            all_num += num
        for label, num in items:
            print('[Info] \t l: {}, n: {}, p: {}%'.format(label, num, round(safe_div(num, all_num) * 100, 2)))

    @staticmethod
    def show_mini_dataset():
        """
        生成mini数据集，包含没有resize的数据
        """
        file_path = os.path.join(DATA_DIR, "files", "4wedu+2wopensource+2.5wnature.labeled-10000.1.out.txt")
        out_html_path = os.path.join(DATA_DIR, "files", "4wedu+2wopensource+2.5wnature.labeled-10000.html")
        print('[Info] 文件: {}'.format(file_path))
        data_lines = read_file(file_path)
        random.seed(47)
        random.shuffle(data_lines)
        data_lines = data_lines[:2000]
        print('[Info] 样本数: {}'.format(len(data_lines)))
        label_str_dict = {"0": "其他", "1": "印刷公式", "2": "印刷文本", "3": "手写公式", "4": "手写文本", "5": "艺术字"}
        label_count_dict = collections.defaultdict(int)

        item_list = []
        for data_idx, data_line in enumerate(data_lines):
            url, label = data_line.split("\t")
            _, img_bgr = download_url_img(url)
            h, w, _ = img_bgr.shape
            if h * w < 3000:
                continue
            shape_str = str(img_bgr.shape)
            label_str = label_str_dict[str(label)]
            label_count_dict[label_str] += 1
            item_list.append([url, label_str, shape_str])
            if data_idx % 10 == 0:
                print("[Info] data_idx: {}".format(data_idx))

        TextLineProcessor.show_num_dict(label_count_dict)
        make_html_page(out_html_path, item_list)
        print('[Info] 写入完成: {}'.format(out_html_path))


def main():
    tlp = TextLineProcessor()
    # tlp.split_train_and_val()
    # tlp.process()
    # tlp.split_labeled_files()
    tlp.show_mini_dataset()


if __name__ == "__main__":
    main()
