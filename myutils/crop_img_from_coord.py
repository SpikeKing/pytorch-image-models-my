#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 16.9.21

核心函数：crop_img_from_coord，多点转换为图像
"""

import copy
import math

import cv2
import numpy as np

SAMPLE_MARGIN = 5

def get_mini_boxes(contour, aspect_ratio=1.7):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    bounding_box = cv2.minAreaRect(contour)
    polygon_area = bounding_box[1][0] * bounding_box[1][1]
    # 当正矩形框和最小外接四边形框的面积接近时，直接采用正矩形框
    if (rect_area - area) / (polygon_area - area + 1e-6) > aspect_ratio:
        points = sorted(tuple(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = np.array((points[index_1], points[index_2],
                        points[index_3], points[index_4]))
        #  p0     p1
        #
        #  p3     p2
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
    else:
        box = np.array(((x, y), (x + w, y), (x + w, y + h), (x, y + h)))
    return box, h, polygon_area


def extract_curly_text(img_, ori_height, ori_width, bottom_curve_pts_, raw_poly_pts_, poly_height_mean_):
    MARGIN = 2
    output_height_ = int(poly_height_mean_ + 2 * MARGIN)
    poly_left_ = np.amin(raw_poly_pts_, 0)[0] - MARGIN
    poly_right_ = np.amax(raw_poly_pts_, 0)[0] + MARGIN

    curve_pts_array_ = np.array(bottom_curve_pts_)
    x_ = curve_pts_array_[:, 0]
    y_ = curve_pts_array_[:, 1]

    bottom_curve_param_ = np.polyfit(x_, y_, deg=3)
    bottom_curve_param_[3] += MARGIN

    x_ = np.clip(np.asarray([poly_left_] +
                            x_.tolist() + [poly_right_]), 0, ori_width)
    bottom_y_ = np.clip(np.poly1d(bottom_curve_param_)
                        (x_).astype(np.int32), 0, ori_height)
    bottom_curve_pts_ = np.asarray([x_, bottom_y_]).T.astype(np.int32)

    curve_length_ = 0
    for i in range(len(bottom_curve_pts_)-1):
        pt_1_ = bottom_curve_pts_[i]
        pt_2_ = bottom_curve_pts_[i+1]
        curve_length_ += math.sqrt(
            pow((pt_1_[0] - pt_2_[0]), 2) + pow((pt_1_[1] - pt_2_[1]), 2))
    curve_length_ = int(curve_length_) + 1

    x_ = np.clip(np.asarray(range(0, curve_length_)) +
                 poly_left_, 0, ori_width - 1)
    bottom_y_ = np.clip(np.poly1d(bottom_curve_param_)(
        x_).astype(np.int32), 0, ori_height - 1)

    all_x_ = x_
    all_y_ = bottom_y_

    top_y_ = None
    for i in range(1, output_height_):
        curve_param_ = copy.deepcopy(bottom_curve_param_)
        curve_param_[3] -= (output_height_ - i)
        y_ = np.clip(np.poly1d(curve_param_)(
            x_).astype(np.int32), 0, ori_height - 1)
        all_y_ = np.hstack((all_y_, y_))
        all_x_ = np.hstack((all_x_, x_))
        if i == 1:
            top_y_ = y_

    bottom_curve_pts_ = np.asarray([x_, bottom_y_]).T.astype(np.int32)
    top_curve_pts_ = np.asarray(
        [np.flip(x_), np.flip(top_y_)]).T.astype(np.int32)

    poly_pts_ = np.concatenate((bottom_curve_pts_, top_curve_pts_), 0)

    align_img_ = img_[all_y_, all_x_].reshape(
        output_height_, curve_length_, 3)
    return align_img_, poly_pts_


def image_affine(src_img, output):
    src_point0 = [int(output[0][0]), int(output[0][1])]
    src_point1 = [int(output[1][0]), int(output[1][1])]
    src_point2 = [int(output[2][0]), int(output[2][1])]
    dst_point0 = [0, 0]
    dst_point1 = [int(output[1][0]) - int(output[0][0]), 0]
    dst_point2 = [int(output[2][0]) - int(output[3][0]),
                  int(output[2][1]) - int(output[1][1])]
    cols = int(int(output[1][0]) - int(output[0][0]))
    rows = max((int(output[3][1]) - int(output[0][1])),
               (int(output[2][1]) - int(output[1][1])))

    pts1 = np.float32([src_point0, src_point1, src_point2])
    pts2 = np.float32([dst_point0, dst_point1, dst_point2])
    M_affine = cv2.getAffineTransform(pts1, pts2)

    img_affine = cv2.warpAffine(src_img, M_affine, (cols, rows))
    return img_affine


def get_curly_text_rect(src_img, output):
    src_point0 = [int(output[0][0]), int(output[0][1])]
    src_point1 = [int(output[1][0]), int(output[1][1])]
    src_point2 = [int(output[2][0]), int(output[2][1])]
    dst_point0 = [0, 0]
    dst_point1 = [int(output[1][0]) - int(output[0][0]), 0]
    dst_point2 = [int(output[2][0]) - int(output[3][0]),
                  int(output[2][1]) - int(output[1][1])]
    cols = int(int(output[1][0]) - int(output[0][0]))
    rows = max((int(output[3][1]) - int(output[0][1])),
               (int(output[2][1]) - int(output[1][1])))

    pts1 = np.float32([src_point0, src_point1, src_point2])
    pts2 = np.float32([dst_point0, dst_point1, dst_point2])
    M_affine = cv2.getAffineTransform(pts1, pts2)

    ret = cv2.warpAffine(src_img, M_affine, (cols, rows))
    return ret


def sample_curve_points(poly_pts, poly_mask, h, w, sample_point_count):
    poly_left_ = np.amin(poly_pts, 0)[0]
    poly_right_ = np.amax(poly_pts, 0)[0]
    poly_width_ = poly_right_ - poly_left_  # 多边形的最大宽度

    pt_x_interval_ = int((poly_width_ - (2*SAMPLE_MARGIN)
                          ) / (sample_point_count + 1))

    ret_bottom_pts_ = []

    for i in range(sample_point_count + 2):
        x_ = poly_left_ + SAMPLE_MARGIN + i * pt_x_interval_
        if x_ >= poly_right_ or x_ >= w:
            break
        non_zeros_ = np.nonzero(poly_mask[:, x_])
        if non_zeros_[0].size == 0:
            break
        y_bottom_ = np.amax(non_zeros_)
        ret_bottom_pts_.append([x_, y_bottom_])

    return ret_bottom_pts_


def is_curly_text(poly_pts, h, w, out_box_region_size_, quad):
    poly_left_ = np.amin(poly_pts, 0)[0]
    poly_right_ = np.amax(poly_pts, 0)[0]

    poly_top_ = np.amin(poly_pts, 0)[1]
    poly_bottom_ = np.amax(poly_pts, 0)[1]
    poly_height_ = abs(poly_bottom_ - poly_top_)

    poly_width_ = poly_right_ - poly_left_  # 多边形的最大宽度
    poly_width_ratio_ = poly_width_ / float(w)

    if poly_width_ratio_ <= 0.3 or poly_height_ < 15:
        return False, [], None, None

    poly_mask_ = np.zeros((h, w), np.uint8)
    cv2.fillPoly(poly_mask_, [poly_pts], 1)

    poly_mask_small_ = get_curly_text_rect(poly_mask_, quad)
    poly_region_size_ = np.sum(poly_mask_small_, dtype=np.uint32)  # 多边形的面积

    region_box_ratio_ = poly_region_size_ / \
        float(out_box_region_size_)  # 多边形面积占最小外接矩阵的比例

    if region_box_ratio_ >= 0.7:
        return False, [], None, None

    poly_region_heights_ = np.sum(poly_mask_small_, 0, np.uint32)
    poly_region_heights_ = poly_region_heights_[
        np.nonzero(poly_region_heights_)]
    poly_height_std_ = np.std(poly_region_heights_)
    poly_height_mean_ = np.max(poly_region_heights_)
    if (poly_height_std_ >= 4.0 or poly_height_mean_ < 15) and region_box_ratio_ >= 0.6:
        return False, [], None, None

    bottom_pts_ = sample_curve_points(poly_pts, poly_mask_, h, w, 12)
    return True, bottom_pts_, poly_height_mean_, poly_mask_


def crop_img_from_coord(coord, img):
    """
    核心函数：多点切图同时转正
    coord: [[591, 4], [589, 16], [596, 27], [640, 24], [656, 21], [653, 12], [652, 4], [628, 0]]
    """
    height, width = img.shape[:2]
    quad, sside, out_box_region_size_ = get_mini_boxes(coord.reshape((-1, 1, 2)))
    is_curly_text_, bottom_curve_pts_, poly_height_mean_, poly_mask_ = is_curly_text(
            coord, height, width, out_box_region_size_, quad)

    if is_curly_text_ and poly_height_mean_ is not None and bottom_curve_pts_ is not None:
        cropped_image, curve_poly_ = extract_curly_text(
            img, height, width, bottom_curve_pts_, coord, poly_height_mean_)
    else:
        cropped_image = image_affine(img, quad)
    return cropped_image


