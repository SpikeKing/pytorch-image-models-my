#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 31.8.21
"""
import os
import sys
import argparse
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.cv_utils import *
from myutils.make_html_page import make_html_page
from myutils.project_utils import *
from x_utils.vpf_sevices import get_vpf_service_np, get_vpf_service


class SampleCleaner(object):
    """
    服务测试函数
    """
    def __init__(self, in_file_or_folder, service_list, out_folder, label_file, num_of_samples, fixed_label):
        self.in_file_or_folder = in_file_or_folder
        self.out_folder = out_folder
        self.service_list = service_list.split("+")
        self.label_file = label_file
        self.num_of_samples = num_of_samples
        self.fixed_label = str(fixed_label)
        print('[Info] 输入文件或文件夹: {}'.format(self.in_file_or_folder))
        print('[Info] 服务: {}'.format(self.service_list))
        print('[Info] 输出文件夹: {}'.format(self.out_folder))
        print('[Info] 标签文件: {}'.format(self.label_file))
        print('[Info] 测试样本数: {}'.format(self.num_of_samples))
        print('[Info] 固定标签值: {}'.format(self.fixed_label))

    @staticmethod
    def save_img_path(img_bgr, img_name, oss_root_dir=""):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        if not oss_root_dir:
            oss_root_dir = "zhengsheng.wcl/imgs-tmp/{}".format(get_current_day_str())
        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    @staticmethod
    def write_results(out_file_format, res):
        """
        写入结果
        """
        out_all_file = out_file_format + "_all.txt"
        out_rgt_file = out_file_format + "_rgt.txt"
        out_err_file = out_file_format + "_err.txt"

        img_url = res[0]
        x_labels = list(set(res[1:]))

        write_line(out_all_file, "\t".join(res))
        print("[Info] x_labels: {}, len: {}".format(x_labels, len(x_labels)))
        if len(x_labels) != 1:
            write_line(out_err_file, img_url)
        else:
            write_line(out_rgt_file, "\t".join(res))

    @staticmethod
    def process_img_path(img_idx, img_path, service_list,  out_file_format):
        img_bgr = cv2.imread(img_path)
        p_label_list = []
        for service in service_list:
            res_dict = get_vpf_service_np(img_np=img_bgr, service_name=service)  # 表格
            p_label = res_dict["data"]["label"]
            p_label = str(p_label)
            p_label_list.append(p_label)

        r_label = str(img_path.split("/")[-2])

        img_name = "{}-{}.jpg".format(get_current_time_str(), time.time())
        img_url = SampleCleaner.save_img_path(img_bgr, img_name)

        res = [img_url, r_label, *p_label_list]
        SampleCleaner.write_results(out_file_format, res)

        x_labels = list(set(p_label_list + [r_label]))
        print('[Info] 处理完成: {}, right: {}'.format(img_idx, len(x_labels) == 1))

    @staticmethod
    def process_img_url(img_idx, img_url, service_list, out_file_format, fixed_label):
        p_label_list = []
        for service in service_list:
            res_dict = get_vpf_service(img_url=img_url, service_name=service)  # 表格
            p_label = res_dict["data"]["label"]
            p_label = str(p_label)
            p_label_list.append(p_label)

        if int(fixed_label) >= 0:
            r_label = str(fixed_label)
        else:
            r_label = str(img_url.split("/")[-2])

        res = [img_url, r_label, *p_label_list]
        SampleCleaner.write_results(out_file_format, res)

        x_labels = list(set(p_label_list + [r_label]))
        if img_idx % 100 == 0:
            print('[Info] 处理完成: {}, right: {}'.format(img_idx, len(x_labels) == 1))

    @staticmethod
    def write_html_results(out_file_format, label_str_list):
        """
        写入结果
        """
        out_all_file = out_file_format + "_all.txt"
        out_rgt_file = out_file_format + "_rgt.txt"
        out_err_file = out_file_format + "_err.txt"
        file_list = [out_all_file, out_rgt_file, out_err_file]

        for file_path in file_list:
            out_html = file_path.replace("txt", "html")
            data_lines = read_file(file_path)
            out_list = []
            max_num = min(200, len(data_lines))
            random.seed(47)
            random.shuffle(data_lines)
            data_lines = data_lines[:max_num]

            for data_line in data_lines:
                items = data_line.split("\t")
                img_url = items[0]
                labels = items[1:]
                out_labels = [label_str_list[int(i)] for i in labels]
                out_list.append([img_url, *out_labels])
            make_html_page(out_html, out_list)
            print('[Info] 处理完成: {}'.format(out_html))

    @staticmethod
    def filter_data_list(data_list, num):
        random.seed(47)
        random.shuffle(data_list)
        xx = min(len(data_list), num)
        data_list = data_list[:xx]
        n = len(data_list)
        return data_list, n

    def process_folder(self):
        time_str = get_current_time_str()
        out_file_format = os.path.join(self.out_folder, "val_{}".format(time_str))

        if self.label_file:
            label_str_list = read_file(self.label_file)
            print('[Info] 标签: {}'.format(label_str_list))
        else:
            label_str_list = [str(x) for x in range(100)]  # 默认数字标签

        pool = Pool(processes=100)
        if not self.in_file_or_folder.endswith("txt"):
            paths_list, names_list = traverse_dir_files(self.in_file_or_folder)
            paths_list, n_sample = SampleCleaner.filter_data_list(paths_list, self.num_of_samples)
            print('[Info] 文件数: {}'.format(n_sample))
            for img_idx, img_path in enumerate(paths_list):
                pool.apply_async(
                    SampleCleaner.process_img_path, (img_idx, img_path, self.service_list, out_file_format))
        else:
            urls = read_file(self.in_file_or_folder)
            urls, n_sample = SampleCleaner.filter_data_list(urls, self.num_of_samples)
            print('[Info] 文件数: {}'.format(n_sample))
            for img_idx, img_url in enumerate(urls):
                # SampleCleaner.process_img_url(img_idx, img_url, self.service_list, out_file_format, self.fixed_label)
                pool.apply_async(
                    SampleCleaner.process_img_url, (img_idx, img_url, self.service_list, out_file_format, self.fixed_label))
        pool.close()
        pool.join()

        data_lines = read_file(out_file_format + "_err.txt")
        n_err = len(data_lines)
        print('[Info] 正确率: {}, {}/{}'.format(safe_div(n_sample - n_err, n_sample), n_err, n_sample))
        SampleCleaner.write_html_results(out_file_format, label_str_list)

        print('[Info] 全部处理完成!')


def parse_args():
    """
    处理脚本参数，支持相对路径
    """
    parser = argparse.ArgumentParser(description='样本清洗')
    parser.add_argument('-i', dest='in_file_or_folder', required=False, help='测试文件或文件夹', type=str)
    parser.add_argument('-s', dest='service_list', required=False, help='服务', type=str)
    parser.add_argument('-o', dest='out_folder', required=False, help='输出文件夹', type=str)
    parser.add_argument('-l', dest='label_file', required=False, help='类别标签文件', type=str)
    parser.add_argument('-n', dest='num_of_samples', required=False, help='测试样本数', type=int, default=-1)
    parser.add_argument('-f', dest='fixed_label', required=False, help='固定标签值', type=int, default=-1)

    args = parser.parse_args()

    arg_in_file_or_folder = args.in_file_or_folder
    print("[Info] 测试文件或文件夹: {}".format(arg_in_file_or_folder))

    arg_service_list = args.service_list
    print("[Info] 服务: {}".format(arg_service_list))

    arg_out_folder = args.out_folder
    print("[Info] 输出文件夹: {}".format(arg_out_folder))
    mkdir_if_not_exist(arg_out_folder)

    arg_label_file = args.label_file
    print("[Info] 标签文件: {}".format(arg_out_folder))

    arg_num_of_samples = args.num_of_samples
    print("[Info] 测试样本数: {}".format(arg_num_of_samples))

    arg_fixed_label = args.fixed_label
    print("[Info] 固定标签值: {}".format(arg_fixed_label))

    return arg_in_file_or_folder, arg_service_list, arg_out_folder, arg_label_file, arg_num_of_samples, arg_fixed_label


def main():
    res_list = parse_args()
    arg_in_file_or_folder, arg_service_list, arg_out_folder, arg_label_file, arg_num_of_samples, arg_fixed_label = res_list
    sc = SampleCleaner(arg_in_file_or_folder, arg_service_list, arg_out_folder,
                       arg_label_file, arg_num_of_samples, arg_fixed_label)
    sc.process_folder()


if __name__ == '__main__':
    main()
