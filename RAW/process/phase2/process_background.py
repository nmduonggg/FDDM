import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def dominant_color(image):
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return dominant

if __name__ == '__main__':

    # background_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/phase2_training_data/Case_2/images/slide-2024-07-26T08-00-29-R1-S2-crop-3251.png'
    background_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/phase2_training_data/Case_2/images/slide-2024-07-26T08-00-29-R1-S2-crop-2505.png'
    object_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/phase2_training_data/Case_2/images/slide-2024-07-26T08-00-29-R1-S2-crop-3253.png'


    bg = cv2.imread(background_path)
    oj = cv2.imread(object_path)
    
    # cv2.imwrite("./bg.png", bg)
    # cv2.imwrite("./oj.png", oj)

    print(dominant_color(bg))
    print(dominant_color(oj))