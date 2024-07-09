import os

import cv2
import numpy as np


def resie_img(img):
    size_tgt = (512, 512)
    img_resized = cv2.resize(img, size_tgt)
    return img_resized


def main():
    dp_src = './data/ICCV/iccv09Data/images/'
    dp_tgt = './data/ICCV/iccv09Data/images_resized/'

    for fn in os.listdir(dp_src):
        print(fn)
        fp_src = dp_src + '/' + fn
        fp_tgt = dp_tgt + '/' + fn
        img = cv2.imread(fp_src)
        img = resie_img(img)
        cv2.imwrite(fp_tgt, img)

    return


if __name__ == '__main__':
    main()
