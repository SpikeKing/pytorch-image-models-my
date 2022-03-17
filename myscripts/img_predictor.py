#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 15.9.21
"""
import os
import sys

import torch
from PIL import Image
from PIL.Image import BICUBIC
from torch.nn import functional as F

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

import timm
from myutils.cv_utils import *
from myutils.project_utils import download_url_img, mkdir_if_not_exist
from root_dir import DATA_DIR
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class ImgPredictor(object):
    """
    预测图像
    """
    def __init__(self, model_path, base_net, num_classes):
        print('[Info] ------ 预测图像 ------')
        self.model_path = model_path
        self.num_classes = num_classes
        self.model, self.transform = self.load_model(self.model_path, base_net, self.num_classes)
        print('[Info] 模型路径: {}'.format(self.model_path))
        print('[Info] base_net: {}'.format(base_net))
        print('[Info] num_classes: {}'.format(num_classes))

    @staticmethod
    def load_model(model_path, base_net, num_classes):
        """
        加载模型
        """
        model = timm.create_model(model_name=base_net, pretrained=False,
                                  checkpoint_path=model_path, num_classes=num_classes)
        if torch.cuda.is_available():
            print('[Info] cuda on!!!')
            model = model.cuda()
        model.eval()

        config_dict = {
            "input_size": (3, 336, 336),
            "interpolation": "bicubic",
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "crop_pct": 1.0  # 不进行Crop
        }

        config = resolve_data_config(config_dict, model=model)
        print("[Info] config: {}".format(config))
        transform = create_transform(**config)

        return model, transform

    @staticmethod
    def img_resize_and_crop(img_pil, size=336, crop_size=336):
        """
        替换transform的resize和crop，提升大约8ms
        """
        w, h = img_pil.size
        if h <= w:
            w = int(w * size / h)
            h = size
        else:
            h = int(h * size / w)
            w = size
        img_pil = img_pil.resize(size=(w, h), resample=BICUBIC)

        width, height = img_pil.size  # Get dimensions
        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = (width + crop_size) / 2
        bottom = (height + crop_size) / 2
        img_pil = img_pil.crop((left, top, right, bottom))

        return img_pil

    @staticmethod
    def img_bgr_norm(img_bgr):
        img_bgr = img_bgr.astype(np.float32)
        for i, x, y in zip((0, 1, 2), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)):
            tmp1 = img_bgr[:, :, i]
            tmp2 = (tmp1 * (1 / 255.0) - x) * (1 / y)
            img_bgr[:, :, i] = tmp2
        img_bgr = img_bgr.transpose((2, 0, 1))
        return img_bgr

    @staticmethod
    def preprocess_img(img_rgb, transform):
        """
        预处理图像
        """
        img_pil = Image.fromarray(img_rgb.astype('uint8'))
        img_tensor = transform(img_pil)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def predict_img(self, img_rgb):
        """
        预测RGB图像
        """
        print('[Info] 预测图像尺寸: {}'.format(img_rgb.shape))
        img_tensor = self.preprocess_img(img_rgb, self.transform)

        # 存储ts模型
        img = torch.zeros(1, 3, 336, 336).to(torch.device('cuda'))
        f = self.model_path.replace('.pth.tar', 'cuda_20220317.ts')  # filename
        with torch.no_grad():
            ts = torch.jit.trace(self.model, img)
        ts.save(f)
        print('[Info] 转换ts完成! ')

        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        print('[Info] 模型输入: {}'.format(img_tensor.shape))
        with torch.no_grad():
            out = self.model(img_tensor)
        print('[Info] 模型结果raw: {}'.format(out))
        probabilities = F.softmax(out[0], dim=0)
        print('[Info] 模型结果: {}'.format(probabilities.shape))
        if self.num_classes >= 5:
            top_n = 5
        else:
            top_n = self.num_classes
        top_prob, top_catid = torch.topk(probabilities, top_n)
        top_catid = list(top_catid.cpu().numpy())
        top_prob = list(top_prob.cpu().numpy())
        top_prob = np.around(top_prob, 4)
        print('[Info] 预测类别: {}'.format(top_catid))
        print('[Info] 预测概率: {}'.format(top_prob))

        return top_catid, top_prob

    def predict_img_path(self, img_path):
        """
        预测图像路径
        """
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        top_catid, top_prob = self.predict_img(img_rgb)
        return top_catid, top_prob

    def predict_img_url(self, img_url):
        """
        预测图像URL
        """
        _, img_bgr = download_url_img(img_url)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        top_catid, top_prob = self.predict_img(img_rgb)
        return top_catid, top_prob

    @staticmethod
    def convert_catid_2_label(catid_list, label_list):
        """
        预测类别id转换为str
        """
        str_list = [label_list[int(ci)] for ci in catid_list]
        return str_list

    def save_pt(self, pt_folder_path, size=336):
        print('[Info] pt存储开始')
        mkdir_if_not_exist(pt_folder_path)
        model_name = self.model_path.split("/")[-1].split(".")[0]
        print('[Info] 模型名称: {}'.format(model_name))
        dummy_shape = (1, 3, size, size)  # 不影响模型
        print('[Info] dummy_shape: {}'.format(dummy_shape))
        if torch.cuda.is_available():
            model_type = "cuda"
        else:
            model_type = "cpu"
        print('[Info] model_type: {}'.format(model_type))
        dummy_input = torch.empty(dummy_shape,
                                  dtype=torch.float32,
                                  device=torch.device(model_type))
        traced = torch.jit.trace(self.model, dummy_input)
        pt_path = os.path.join(pt_folder_path, "{}_{}.pt".format(model_name, model_type))
        traced.save(pt_path)

        with torch.no_grad():
            standard_out = self.model(dummy_input)
        print('[Info] standard_out: {}'.format(standard_out))

        reload_script = torch.jit.load(pt_path)
        with torch.no_grad():
            script_output = reload_script(dummy_input)
        print('[Info] script_output: {}'.format(script_output))
        print('[Info] 验证 is equal: {}'.format(F.l1_loss(standard_out, script_output)))

        print('[Info] 存储完成: {}'.format(pt_path))
        return pt_path


def main_4_doc_clz():
    img_path = os.path.join(DATA_DIR, "document_dataset_mini", "train", "000", "train_040000_000.jpg")

    case_url = "http://quark-cv-data.oss-cn-hangzhou.aliyuncs.com/gaoyan/project/gt_imaage_for_biaozhu3/" \
               "O1CN0100fHnP21yK9SLVNC9_!!6000000007053-0-quark.jpg"

    model_path = os.path.join(DATA_DIR, "models", "model_best_c2_20210915.pth.tar")
    base_net = "resnet50"
    num_classes = 2
    label_list = ["纸质文档", "其他"]

    # show_img_bgr(cv2.imread(img_path))

    me = ImgPredictor(model_path, base_net, num_classes)
    # top5_catid, top5_prob = me.predict_img_path(img_path)
    top5_catid, top5_prob = me.predict_img_url(case_url)
    top5_cat = me.convert_catid_2_label(top5_catid, label_list)
    print('[Info] 预测类别: {}'.format(top5_cat))
    print('[Info] 预测概率: {}'.format(top5_prob))
    # me.save_pt(os.path.join(DATA_DIR, "pt_models"))  # 存储PT模型


def main_4_text_line_clz():
    img_path = os.path.join(DATA_DIR, "datasets", "text_line_v1_200w", "train", "000", "train_0000000.jpg")  # 其他
    # img_path = os.path.join(DATA_DIR, "datasets", "text_line_v1_200w", "train", "001", "train_0000000.jpg")  # 印刷公式
    # img_path = os.path.join(DATA_DIR, "datasets", "text_line_v1_200w", "train", "002", "train_0000000.jpg")  # 印刷文本
    # img_path = os.path.join(DATA_DIR, "datasets", "text_line_v1_200w", "train", "003", "train_0000000.jpg")  # 手写公式
    # img_path = os.path.join(DATA_DIR, "datasets", "text_line_v1_200w", "train", "004", "train_0000000.jpg")  # 手写文本
    # img_path = os.path.join(DATA_DIR, "document_dataset_mini", "train", "005", "train_080000_005.jpg")

    model_path = os.path.join(DATA_DIR, "models", "model_best_tlc_c5_20210920.pth.tar")
    base_net = "resnet50"
    num_classes = 5
    label_list = ["其他", "印刷公式", "印刷文本", "手写公式", "手写文本"]

    # show_img_bgr(cv2.imread(img_path))

    me = ImgPredictor(model_path, base_net, num_classes)
    top5_catid, top5_prob = me.predict_img_path(img_path)
    # top5_catid, top5_prob = me.predict_img_url(case_url)
    top5_cat = me.convert_catid_2_label(top5_catid, label_list)
    print('[Info] 预测类别: {}'.format(top5_cat))
    print('[Info] 预测概率: {}'.format(top5_prob))
    # me.save_pt(os.path.join(DATA_DIR, "pt_models"))  # 存储PT模型


def main_4_fonts_clz():
    img_url = "http://sm-compute-public-data.oss-cn-beijing.aliyuncs.com/zhengsheng.wcl/document_style_manuel/datasets/regions-20220315/3ba0d6299db446529e959020aa62ed40_pos_000000.jpg"
    img_url = "http://sm-compute-public-data.oss-cn-beijing.aliyuncs.com/zhengsheng.wcl/document_style_manuel/datasets/regions-20220315/00185d07707c40c5a7387e4445c6c94a_pos_000003.jpg"
    img_url = "http://sm-compute-public-data.oss-cn-beijing.aliyuncs.com/zhengsheng.wcl/document_style_manuel/datasets/regions-20220315/ffbb81ce0fe5454e96112b6e2bfaaf4e_neg_000002.jpg"

    img_path = os.path.join(DATA_DIR, "datasets/ch_en_line_v1/train/000/train_00000000_000.jpg")
    model_path = os.path.join(DATA_DIR, "models", "regions_clz_best_20220316_checkpoint-52.pth.tar")
    base_net = "efficientnet_b2"
    num_classes = 3
    me = ImgPredictor(model_path, base_net, num_classes)
    top5_catid, top5_prob = me.predict_img_url(img_url)
    print('[Info] 预测类别: {}'.format(top5_catid))
    print('[Info] 预测概率: {}'.format(top5_prob))


if __name__ == '__main__':
    # main_4_doc_clz()
    main_4_fonts_clz()
