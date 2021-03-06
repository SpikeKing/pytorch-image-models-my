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


class ServiceTester(object):
    """
    服务测试函数
    """
    def __init__(self, in_file_or_folder, service, out_folder, label_file, num_of_samples):
        self.in_file_or_folder = in_file_or_folder
        self.out_folder = out_folder
        self.service = service
        self.label_file = label_file
        self.num_of_samples = num_of_samples
        print('[Info] 输入文件或文件夹: {}'.format(self.in_file_or_folder))
        print('[Info] 服务: {}'.format(self.service))
        print('[Info] 输出文件夹: {}'.format(self.out_folder))
        print('[Info] 标签文件: {}'.format(self.label_file))
        print('[Info] 测试样本数: {}'.format(self.num_of_samples))

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

        img_url, r_label, p_label = res

        write_line(out_all_file, "{}\t{}\t{}".format(*res))
        if p_label != r_label:
            write_line(out_err_file, "{}\t{}\t{}".format(*res))
        else:
            write_line(out_rgt_file, "{}\t{}\t{}".format(*res))

    @staticmethod
    def process_img_path(img_idx, img_path, service,  out_file_format):
        img_bgr = cv2.imread(img_path)
        res_dict = get_vpf_service_np(img_np=img_bgr, service_name=service)  # 表格
        p_label = res_dict["data"]["label"]
        p_label = int(p_label)
        r_label = int(img_path.split("/")[-2])

        img_name = "{}-{}.jpg".format(get_current_time_str(), time.time())
        img_url = ServiceTester.save_img_path(img_bgr, img_name)

        res = [img_url, r_label, p_label]
        ServiceTester.write_results(out_file_format, res)

        print('[Info] 处理完成: {}, right: {}'.format(img_idx, p_label == r_label))

    @staticmethod
    def process_img_url(img_idx, img_url, service, out_file_format, img_label):
        res_dict = get_vpf_service(img_url=img_url, service_name=service)  # 表格
        p_label = res_dict["data"]["label"]
        p_label = int(p_label)
        r_label = int(img_label)

        res = [img_url, r_label, p_label]
        ServiceTester.write_results(out_file_format, res)

        print('[Info] 处理完成: {}, right: {}'.format(img_idx, p_label == r_label))

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
            for data_line in data_lines:
                img_url, r_label, p_label = data_line.split("\t")
                out_list.append([img_url, label_str_list[int(r_label)], label_str_list[int(p_label)]])
            make_html_page(out_html, out_list)
            print('[Info] 处理完成: {}'.format(out_html))

    def process_folder(self):
        time_str = get_current_time_str()
        out_file_format = os.path.join(self.out_folder, "val_{}".format(time_str))

        if self.label_file:
            label_str_list = read_file(self.label_file)
            print('[Info] 标签: {}'.format(label_str_list))
        else:
            label_str_list = ["印刷文字", "手写文字", "艺术字", "其他"]

        def filter_data_list(data_list, num):
            random.seed(47)
            random.shuffle(data_list)
            xx = min(len(data_list), num)
            data_list = data_list[:xx]
            n = len(data_list)
            return data_list, n

        pool = Pool(processes=100)
        if not self.in_file_or_folder.endswith("txt"):
            paths_list, names_list = traverse_dir_files(self.in_file_or_folder)
            paths_list, n_sample = filter_data_list(paths_list, self.num_of_samples)
            print('[Info] 文件数: {}'.format(n_sample))
            for img_idx, img_path in enumerate(paths_list):
                pool.apply_async(
                    ServiceTester.process_img_path, (img_idx, img_path, self.service, out_file_format))
        else:
            data_lines = read_file(self.in_file_or_folder)
            urls, labels = [], []
            for data_line in data_lines:
                url, label = data_line.split("\t")
                urls.append(url)
                labels.append(label)
            # urls, n_sample = filter_data_list(urls, self.num_of_samples)
            n_sample = len(urls)
            print('[Info] 文件数: {}'.format(n_sample))
            for img_idx, (img_url, img_label) in enumerate(zip(urls, labels)):
                pool.apply_async(
                    ServiceTester.process_img_url, (img_idx, img_url, self.service, out_file_format, img_label))
        pool.close()
        pool.join()

        data_lines = read_file(out_file_format + "_err.txt")
        n_err = len(data_lines)
        print('[Info] 正确率: {}, {}/{}'.format(safe_div(n_sample - n_err, n_sample), n_err, n_sample))
        ServiceTester.write_html_results(out_file_format, label_str_list)

        print('[Info] 全部处理完成!')


def parse_args():
    """
    处理脚本参数，支持相对路径
    """
    parser = argparse.ArgumentParser(description='服务测试')
    parser.add_argument('-i', dest='in_file_or_folder', required=False, help='测试文件或文件夹', type=str)
    parser.add_argument('-s', dest='service', required=False, help='服务', type=str)
    parser.add_argument('-o', dest='out_folder', required=False, help='输出文件夹', type=str)
    parser.add_argument('-l', dest='label_file', required=False, help='类别标签文件', type=str)
    parser.add_argument('-n', dest='num_of_samples', required=False, help='测试样本数', type=int, default=200)

    args = parser.parse_args()

    arg_in_file_or_folder = args.in_file_or_folder
    print("[Info] 测试文件或文件夹: {}".format(arg_in_file_or_folder))

    arg_service = args.service
    print("[Info] 服务: {}".format(arg_service))

    arg_out_folder = args.out_folder
    print("[Info] 输出文件夹: {}".format(arg_out_folder))
    mkdir_if_not_exist(arg_out_folder)

    arg_label_file = args.label_file
    print("[Info] 标签文件: {}".format(arg_out_folder))

    arg_num_of_samples = args.num_of_samples
    print("[Info] 测试样本数: {}".format(arg_num_of_samples))


    return arg_in_file_or_folder, arg_service, arg_out_folder, arg_label_file, arg_num_of_samples


def main():
    res_list = parse_args()
    arg_in_file_or_folder, arg_service, arg_out_folder, arg_label_file, arg_num_of_samples = res_list
    st = ServiceTester(arg_in_file_or_folder, arg_service, arg_out_folder, arg_label_file, arg_num_of_samples)
    st.process_folder()


if __name__ == '__main__':
    main()
