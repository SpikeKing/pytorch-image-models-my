#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 15.9.21
"""
import argparse
import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import DATA_DIR
from myscripts.img_predictor import ImgPredictor


def parse_args():
    """
    处理脚本参数
    """
    parser = argparse.ArgumentParser(description='PyTorch模型转换PT模型')
    parser.add_argument('-m', dest='model_path', required=True, help='模型路径', type=str)
    parser.add_argument('-n', dest='base_net', required=False, help='basenet', type=str, default="resnet50")
    parser.add_argument('-c', dest='num_classes', required=False, help='类别个数', type=int, default=2)
    parser.add_argument('-o', dest='out_dir', required=False, help='输出文件夹', type=str,
                        default=os.path.join(DATA_DIR, "pt_models"))

    args = parser.parse_args()

    arg_model_path = args.model_path
    print("[Info] 模型路径: {}".format(arg_model_path))

    arg_base_net = args.base_net
    print("[Info] basenet: {}".format(arg_base_net))

    arg_num_classes = args.num_classes
    print("[Info] 类别数: {}".format(arg_num_classes))

    arg_out_dir = args.out_dir
    print("[Info] 输出文件夹: {}".format(arg_out_dir))

    return arg_model_path, arg_base_net, arg_num_classes, arg_out_dir


def main():
    """
    入口函数
    """
    print('[Info] ' + "-" * 100)
    print('[Info] 转换PT模型开始')
    arg_model_path, arg_base_net, arg_num_classes, arg_out_dir = parse_args()
    me = ImgPredictor(arg_model_path, arg_base_net, arg_num_classes)
    pt_path = me.save_pt(arg_out_dir)  # 存储PT模型
    print('[Info] 存储完成: {}'.format(pt_path))
    print('[Info] ' + "-" * 100)


if __name__ == "__main__":
    main()
