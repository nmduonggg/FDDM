'''
Filter the blank patches out of a given dataset dict
'''

import os
import json 
import cv2
import numpy as np
from tqdm import tqdm

def get_case_dict(path):
    with open(path, 'r') as f:
        case_dict = json.load(f)
    return case_dict

def open_img(path):
    img = cv2.imread(path)[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_classes_by_threshold(image):
    # Create masks for pixels that are closer to green or pink
    # Initialize the output image with the original image
    
    target_colors = [
        [255, 255, 255],
        [0, 128, 0],
        [255, 143, 204],
        [255, 0, 0],
        [0, 0, 0],
        [165, 42, 42],
        [0, 0, 255]]
    tolerance = 10
    
    classes = []
    for idx, color in enumerate(target_colors):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        # classes[idx] = (np.sum(mask) > 0)
        classes.append(mask.mean())
    return np.argmax(np.array(classes))

def print_statistic_class(class_dict):
    for cl, images in class_dict.items():
        print(f"class {cl}: {len(images)}", end = "| ")

if __name__=="__main__":
    label_dir = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/training_data_256/labels'
    image_dir = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/training_data_256/images'
    # case_dict_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/case_dict_256.json'
    # class_dict_outpath = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/class_dict_256_rm_confused_case.json'
    dataset_dict_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/dataset_split_256_by_class_rm_confused_case.json'
    new_dataset_dict_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/dataset_dict_256_rm_confused_case_rm_blank.json'
    
    # class_dict = {}
    
    dataset_dict = get_case_dict(dataset_dict_path)
    skip_counts = 0
    new_dataset_dict = {}
    for mode, indices in dataset_dict.items():
        if "last_index" in mode: continue
        print(f"Mode: {mode}")
        for idx in tqdm(indices, total=len(indices)):
            label_path = os.path.join(label_dir, f"gt_{idx}.png")
            img_path = os.path.join(image_dir, f"patch_{idx}.png")
            img = open_img(img_path)
            label = open_img(label_path)
            label = get_classes_by_threshold(label)
            
            skip_condition = (label != 0) and ( abs(np.mean(img) - 255) < 50 )  # remove confused blank except for label==0
            if skip_condition: 
                skip_counts += 1
                continue
            
            new_dataset_dict[mode] = new_dataset_dict.get(mode, []) + [idx]
                
        print_statistic_class(new_dataset_dict)
        print(skip_counts)
                
    print_statistic_class(new_dataset_dict)
    print(skip_counts)
    
    with open(new_dataset_dict_path, "w") as outfile: 
        json.dump(new_dataset_dict, outfile)
            