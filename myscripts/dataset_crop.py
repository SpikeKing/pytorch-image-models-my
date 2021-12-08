#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 6.12.21
"""

import os
import math
import random
from PIL import Image
from myutils.cv_utils import *
from root_dir import DATA_DIR


def get_params(img, scale, ratio):
    """Get parameters for ``crop`` for a random sized crop.

    Args:
        img (PIL Image): Image to be cropped.
        scale (tuple): range of size of the origin size cropped
        ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
    """
    area = img.size[0] * img.size[1]

    for attempt in range(10):
        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if w <= img.size[0] and h <= img.size[1]:
            i = random.randint(0, img.size[1] - h)
            j = random.randint(0, img.size[0] - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = img.size[0] / img.size[1]
    if in_ratio < min(ratio):
        w = img.size[0]
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = img.size[1]
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = img.size[0]
        h = img.size[1]
    i = (img.size[1] - h) // 2
    j = (img.size[0] - w) // 2
    return i, j, h, w


def main():
    img = os.path.join(DATA_DIR, "00057e0d-51f4-4879-8116-cacb810482ce.jpg")
    img_pil = Image.open(img)
    top, left, height, width = get_params(img_pil, (0.08, 1.0), (3. / 4., 4. / 3.))
    print(img_pil.size)
    img_pil_crop = img_pil.crop((left, top, left+width, top+height))
    print(img_pil_crop.size)
    img_pil_crop.save("tmp.jpg")


if __name__ == '__main__':
    main()


