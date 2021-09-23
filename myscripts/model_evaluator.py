#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 15.9.21
"""

from multiprocessing.pool import Pool

import torch
from PIL import Image

import timm
from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class ModelEvaluator(object):
    def __init__(self):
        # 文本行分类
        # self.model_path = os.path.join(DATA_DIR, "models", "model_best_tlc_c5_20210920.pth.tar")
        # self.label_str_list = ["其他", "印刷公式", "印刷文本", "手写公式", "手写文本"]
        # self.file_path = os.path.join(DATA_DIR, "files", "common_full_out_v1_200w.txt")
        # self.out_file_path = os.path.join(DATA_DIR, "files", "out_predict.{}.txt".format(get_current_time_str()))

        # 文本行分类
        self.model_path = os.path.join(DATA_DIR, "models", "model_best_c2_20210922.pth.tar")
        self.label_str_list = ["文档", "非文档"]
        self.file_path = os.path.join(DATA_DIR, "files", "out_labeled_urls.txt")
        self.out_file_path = os.path.join(DATA_DIR, "files", "out_predict.{}.txt".format(get_current_time_str()))
        self.num_clz = 2

        self.model, self.transform = self.load_model(self.model_path, self.num_clz)

    @staticmethod
    def load_model(model_path, num_clz):
        model = timm.create_model('resnet50', pretrained=True, checkpoint_path=model_path, num_classes=num_clz)
        if torch.cuda.is_available():
            print('[Info] cuda on!!!')
            model = model.cuda()
        model.eval()
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        return model, transform

    @staticmethod
    def preprocess_img(img_rgb, transform):
        img_pil = Image.fromarray(img_rgb.astype('uint8')).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0)  # transform and add batch dimension
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        return img_tensor

    def predict_img(self, img_rgb):
        img_tensor = self.preprocess_img(img_rgb, self.transform)
        with torch.no_grad():
            out = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        # print("[Info] 输出尺寸: {}".format(probabilities.shape))
        top5_prob, top5_catid = torch.topk(probabilities, 2)
        label_list = list(top5_catid.cpu().numpy())
        # print(label_list)
        # for i in range(top5_prob.size(0)):
        #     print("[Info] 类别: {}, 概率: {}".format(self.label_str_list[top5_catid[i]], top5_prob[i].item()))
        return label_list

    @staticmethod
    def process_line(img_idx, img_url, label, out_file_path, me):
        print('[Info] 处理开始: {}'.format(img_url))
        if label == 0:
            label = 0
        else:
            label = 1
        _, img_bgr = download_url_img(img_url)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        label_list = me.predict_img(img_rgb)
        label_list = [str(i) for i in label_list]
        out_line = "{}\t{}".format(label, ",".join(label_list))
        write_line(out_file_path, out_line)
        print('[Info] 处理完成: {}'.format(img_idx))

    def evaluate_dataset(self):
        data_lines = read_file(self.file_path)
        print('[Info] 处理文件: {}'.format(self.file_path))
        random.seed(47)
        random.shuffle(data_lines)
        data_lines = data_lines[:1000]
        print('[Info] 样本数: {}'.format(len(data_lines)))
        pool = Pool(processes=1)
        me = ModelEvaluator()
        for img_idx, data_line in enumerate(data_lines):
            img_url, label = data_line.split("\t")
            ModelEvaluator.process_line(img_idx, img_url, label, self.out_file_path, me)
            # pool.apply_async(ModelEvaluator.process_line, (img_idx, img_url, label, self.out_file_path, me))
        pool.close()
        pool.join()
        print('[Info] 写入完成: {}'.format(self.out_file_path))

        data_lines = read_file(self.out_file_path)
        l_dict = collections.defaultdict(int)
        top1, top2, top3 = collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int)
        for data_line in data_lines:
            l, l_list = data_line.split("\t")
            l_list = l_list.split(",")
            l_dict[l] += 1
            if l == l_list[0]:
                top1[l] += 1
            if l in l_list[:2]:
                top2[l] += 1
            if l in l_list[:3]:
                top3[l] += 1
        print("[Info] l_dict: {}".format(l_dict))
        for l in l_dict.keys():
            print("[Info]" + "-" * 100)
            print("[Info]\t {}".format(self.label_str_list[int(l)]))
            total_n = l_dict[l]
            top1_n = top1[l]
            top2_n = top2[l]
            top3_n = top3[l]
            acc_top1 = round(safe_div(top1_n, total_n) * 100, 4)
            acc_top2 = round(safe_div(top2_n, total_n) * 100, 4)
            acc_top3 = round(safe_div(top3_n, total_n) * 100, 4)
            print("[Info]\t top1: {}%, {}/{}".format(acc_top1, top1_n, total_n))
            print("[Info]\t top2: {}%, {}/{}".format(acc_top2, top2_n, total_n))
            print("[Info]\t top3: {}%, {}/{}".format(acc_top3, top3_n, total_n))


def main():
    # img_path = os.path.join(DATA_DIR, "document_dataset_mini", "train", "000", "train_040000_000.jpg")
    # img_path = os.path.join(DATA_DIR, "document_dataset_mini", "train", "001", "train_060000_001.jpg")
    # img_path = os.path.join(DATA_DIR, "document_dataset_mini", "train", "002", "train_020000_002.jpg")
    # img_path = os.path.join(DATA_DIR, "document_dataset_mini", "train", "003", "train_100000_003.jpg")
    # img_path = os.path.join(DATA_DIR, "document_dataset_mini", "train", "004", "train_000000_004.jpg")
    # img_path = os.path.join(DATA_DIR, "document_dataset_mini", "train", "005", "train_080000_005.jpg")
    # img_bgr = cv2.imread(img_path)
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    me = ModelEvaluator()
    # me.predict_img(img_rgb)
    me.evaluate_dataset()


if __name__ == '__main__':
    main()
