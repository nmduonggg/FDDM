'''
Scan through the case_dict to see the class distribution in each case, group and divide them into class_dict
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
    label_dir = '/home/user01/aiotlab/nmduong/BoneTumor/RAW/REAL_WSIs/training_data_256_case68/labels'
    image_dir = '/home/user01/aiotlab/nmduong/BoneTumor/RAW/REAL_WSIs/training_data_256_case68/images'
    case_dict_path = '/home/user01/aiotlab/nmduong/BoneTumor/RAW/REAL_WSIs/case_dict_256_case68.json'
    # class_dict_inpath = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/class_dict_256_rm_confused_case.json'
    class_dict_outpath = '/home/user01/aiotlab/nmduong/BoneTumor/RAW/REAL_WSIs/class_dict_256_case68.json'
    class_dict = {}
    # class_dict = get_case_dict(class_dict_inpath)
    
    case_dict = get_case_dict(case_dict_path)
    skip_counts = 0
    for case, indices in case_dict.items():
        # if case!="9": continue  # only add case 9
        
        if "last_index" in case: continue
        print(f"Case: {case}")
        for idx in tqdm(indices, total=len(indices)):
            label_path = os.path.join(label_dir, f"gt_{idx}.png")
            img_path = os.path.join(image_dir, f"patch_{idx}.png")
            img = open_img(img_path)
            label = open_img(label_path)
            label = get_classes_by_threshold(label)
            
            # if label not in [3, 4, 5]: 
            #     continue
            
            non_blank_img_condition = (label==0) and ((abs(np.mean(img) - 255) > 5) or (abs(np.mean(img[:, :, -1]) - 255) > 5))
            blank_img_condition = (label != 0) and ((abs(np.mean(img) - 255) <= 5) or (abs(np.mean(img[:, :, -1]) - 255) <= 5))
            
            skip_condition = non_blank_img_condition or blank_img_condition
            if skip_condition: 
                skip_counts += 1
                continue
            
            class_dict[int(label)] = class_dict.get(int(label), []) + [idx]
                
        print_statistic_class(class_dict)
        print(skip_counts)
                
    print_statistic_class(class_dict)
    print(skip_counts)
    
    with open(class_dict_outpath, "w") as outfile: 
        json.dump(class_dict, outfile)
            