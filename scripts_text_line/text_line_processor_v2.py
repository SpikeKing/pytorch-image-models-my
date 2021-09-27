#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 27.9.21
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


class TextLineProcessorV2(object):
    """
    解析原始数据
    1. 拆分文档和自然场景
    2. 提取框并且上传图像，过滤小于200 * 200的图像
    3. 创建可以直接访问图像的数据集
    """

    def __init__(self):
        folder_path = os.path.join(DATA_DIR, "files_v2")
        self.raw_file = os.path.join(folder_path, "4wedu+2wopensource+2.5wnature.txt")
        self.cat_file = os.path.join(folder_path, "文字检测_识别文字检测数据清洗_.txt")
        self.raw_nat_file = os.path.join(folder_path, "text_line_raw_nat.txt")
        self.raw_doc_file = os.path.join(folder_path, "text_line_raw_doc.txt")
        self.nat_dataset_file = os.path.join(folder_path, "text_line_nat_dataset.{}.txt".format(get_current_time_str()))
        self.doc_dataset_file = os.path.join(folder_path, "text_line_doc_dataset.{}.txt".format(get_current_time_str()))

    @staticmethod
    def save_img_path(img_bgr, img_name, oss_root_dir=""):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        if not oss_root_dir:
            oss_root_dir = "zhengsheng.wcl/Text-Line-Clz/datasets/v2/{}".format(get_current_day_str())
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
        if avg_x * avg_y < 200 * 200:
            return []

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
    def process_line(data_idx, data_type, data_line, out_file_path):
        data = eval(data_line.strip())
        url = data["url"]
        labels = data['label']
        coords = data['coord']
        _, img_bgr = download_url_img(url)

        # 处理正例
        for i in range(len(labels)):
            label = labels[i]
            if label in [1, 2, 3, 4, 5]:
                coord = np.array(coords[i])
                crop_img = crop_img_from_coord(coord, img_bgr)
                crop_img_name = "{}_{}_{}_{}.jpg".format(data_type, str(data_idx).zfill(7), str(i), str(label))
                h, w, _ = crop_img.shape
                if h * w < 100 * 100:
                    continue
                crop_img_url = TextLineProcessorV2.save_img_path(crop_img, crop_img_name)
                out_dict = {
                    "ori_img_url": url,
                    "coord_idx": i,
                    "crop_img_url": crop_img_url,
                    "label": label
                }
                write_line(out_file_path, json.dumps(out_dict))

        # 处理负例
        neg_boxes = TextLineProcessorV2.get_negative_box(img_bgr, coords)
        for i, neg_box in enumerate(neg_boxes):
            neg_label = 0  # 负例标签是0
            crop_img = get_cropped_patch(img_bgr, neg_box)
            crop_img_name = "{}_{}_{}_{}.jpg".format(data_type, str(data_idx).zfill(7), str(i + len(labels)), str(neg_label))
            crop_img_url = TextLineProcessorV2.save_img_path(crop_img, crop_img_name)
            out_dict = {
                "ori_img_url": url,
                "coord_idx": -1,
                "crop_img_url": crop_img_url,
                "label": neg_label
            }
            write_line(out_file_path, json.dumps(out_dict))

        if data_idx % 1000 == 0:
            print('[Info] 处理完成: {}'.format(data_idx))

    @staticmethod
    def process_line_try(data_idx, data_type, data_line, out_file_path):
        try:
            TextLineProcessorV2.process_line(data_idx, data_type, data_line, out_file_path)
        except Exception as e:
            print('[Error] data_idx: {}'.format(data_idx))
            print('[Error] e: {}'.format(e))

    def split_cat_files(self):
        """
        拆分类别文档
        """
        print("[Info] 原始文件: {}".format(self.raw_file))
        print("[Info] 类别文件: {}".format(self.cat_file))
        raw_data_lines = read_file(self.raw_file)
        print("[Info] 样本数: {}".format(len(raw_data_lines)))
        cat_data_lines = read_file(self.cat_file)
        print("[Info] 样本数: {}".format(len(cat_data_lines)))

        raw_dict = dict()
        for data_idx, raw_data in enumerate(raw_data_lines):
            data_dict = eval(raw_data)
            url = data_dict["url"]
            raw_dict[url] = raw_data
            if data_idx % 10000 == 0:
                print('[Info] \tdata_idx: {}'.format(data_idx))
        print('[Info] 原始样本数: {}'.format(len(raw_dict.keys())))

        nat_list = []
        doc_list = []
        for data_idx, cat_data in enumerate(cat_data_lines):
            data_dict = json.loads(cat_data)
            url = data_dict["url"]
            label = data_dict["cate"]
            if url in raw_dict.keys():
                if label == "自然":
                    raw_line = raw_dict[url]
                    nat_list.append(raw_line)
                elif label == "文档":
                    raw_line = raw_dict[url]
                    doc_list.append(raw_line)
            if data_idx % 10000 == 0:
                print('[Info] \tdata_idx: {}'.format(data_idx))
        print('[Info] 原始样本数: {}'.format(len(raw_dict.keys())))

        write_list_to_file(self.raw_nat_file, nat_list)
        write_list_to_file(self.raw_doc_file, doc_list)
        print('[Info] 处理完成!')

    def parse_raw_file(self):
        # print('[Info] 处理文件: {}'.format(self.raw_nat_file))
        print('[Info] 处理文件: {}'.format(self.raw_doc_file))
        # nat_type = "nat"
        doc_type = "doc"
        # nat_data_lines = read_file(self.raw_nat_file)
        # print('[Info] nat样本数: {}'.format(len(nat_data_lines)))
        doc_data_lines = read_file(self.raw_doc_file)
        print('[Info] doc样本数: {}'.format(len(doc_data_lines)))

        pool = Pool(processes=40)
        # for data_idx, data_line in enumerate(nat_data_lines):
            # pool.apply_async(TextLineProcessorV2.process_line_try, (data_idx, "nat", data_line, self.nat_dataset_file))
        for data_idx, data_line in enumerate(doc_data_lines):
            pool.apply_async(TextLineProcessorV2.process_line_try, (data_idx, "doc", data_line, self.doc_dataset_file))
        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(self.nat_dataset_file))
        # print('[Info] 处理完成: {}'.format(self.doc_dataset_file))


def main():
    tlp2 = TextLineProcessorV2()
    # tlp2.split_cat_files()  # 拆分自然场景和文档
    tlp2.parse_raw_file()  # 处理原始文件，上传图像


if __name__ == '__main__':
    main()
