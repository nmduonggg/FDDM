"""
Cut images and labels in folder Case_x into patches 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

def open_img(img_path):
    out = cv2.imread(img_path)[:, :, :3]
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.uint8)
    return out

def apply_threshold_mapping(image):
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
    tolerance = 50
    
    output = np.ones_like(image)*255 # 2D only
    masks = []
    for idx, color in enumerate(target_colors):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        output[mask] = color
        # output[mask] = idx

    return output

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
    tolerance = 50
    
    classes = []
    for idx, color in enumerate(target_colors):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        # classes[idx] = (np.sum(mask) > 0)
        classes.append(mask.mean())
    return np.argmax(np.array(classes))

def cut_image(image,  
                crop_sz=224, step=200,
                outdir = "./RAW/REAL_WSIs/training_data/", gt=False,):
    global im_index, gt_index
    global skip_indices
    global class_dict
    
    image_outdir = os.path.join(outdir, "images")
    label_outdir = os.path.join(outdir, "labels")
    # print("Create folder: ", image_outdir, label_outdir)
    os.makedirs(image_outdir, exist_ok=True)
    os.makedirs(label_outdir, exist_ok=True)
    
    h, w, c = image.shape
    # cut WSIs into tiles
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    num_h = 0
    
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            crop_img = image[x:x + crop_sz, y:y + crop_sz, :]
            
            if gt:
                crop_img = apply_threshold_mapping(crop_img)    # clean the color
                if np.mean(crop_img)==255:
                    skip_indices.append([x, y])
                    continue
                plt.imsave(os.path.join(label_outdir, f"gt_{gt_index}.png"), crop_img.astype(np.uint8))
                label_idx = get_classes_by_threshold(crop_img)
                # class_dict[int(label_idx)] = class_dict.get(int(label_idx), [0]) + [gt_index]
                class_dict[str(label_idx)] += [gt_index]
                gt_index += 1
                
            else:
                if [x, y] in skip_indices:
                    continue
                plt.imsave(os.path.join(image_outdir, f"patch_{im_index}.png"), crop_img.astype(np.uint8))
                im_index += 1
            
    h=x + crop_sz
    w=y + crop_sz


if __name__ == "__main__":
    
    ### CONFIGURATION
    done_track = [1, 2, 3, 9, 5, 7, 10, 4]  # 6, 8 for testing
    cases = []
    image_folder = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/RAW_DATA/images"
    label_folder = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/RAW_DATA/labels"
    case_dict_inpath = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/case_dict_256_bkup.json"
    case_dict_outpath = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/case_dict_256.json"
    class_dict_inpath = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/class_dict_256_bkup.json"
    class_dict_outpath = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/class_dict_256.json"
    
    try:
        with open(case_dict_inpath, 'r') as f:
            case_dict = json.load(f)
    except:
        case_dict = {}  # keep track which patches belong to which case
        print("No preload case dict founda, use empty dict")
        
    try:
        with open(class_dict_inpath, 'r') as f:
            class_dict = json.load(f)
    except:
        class_dict = {}  # keep track which patches belong to which case
        print("No preload class dict founda, use empty dict")
        
    # print(class_dict.keys())

    
    crop_sz = 256
    step = 256
    im_index = case_dict.get('last_index', 0)
    gt_index = case_dict.get('last_index', 0)
    
    print(f"Start from {im_index}")
    
    # Crop images and labels separatedly
    for tcase in cases:
        print("Processing case: ", tcase)
        case_label_folder = os.path.join(label_folder, f"Case_{tcase}")
        for label_name in os.listdir(case_label_folder):
            if 'slide-' not in label_name: continue
            print("Processing: ", label_name)
            
            # skip indices of mask -> store to process in image
            skip_indices = []
            upsample=False
            
            if "x8" in label_name:
                img_name = label_name.split("-x8")[0] + '.png'
                upsample = True
            else:
                img_name = label_name.split("-labels")[0] + '.png'
                
            image_path = os.path.join(image_folder, f"Case_{tcase}", img_name)
            image = open_img(image_path)
            image_shape = image.shape
            print(f"image shape: {image.shape}")
            del image   # free mem
            
            label = open_img(os.path.join(case_label_folder, label_name))
            label = cv2.resize(label, (image_shape[1], image_shape[0]), cv2.INTER_NEAREST_EXACT)
            # label = apply_threshold_mapping(label)
                
            print(f"label shape: {label.shape}")
            
            cut_image(label,
                crop_sz=crop_sz, step=step,
                outdir = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/training_data_256",
                gt=True)
            
            # print(label.shape, image.shape)
            del label
            
            image = open_img(image_path)
            
            old_index = im_index
            cut_image(image,
                    crop_sz=crop_sz, step=step,
                    outdir = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/training_data_256",
                    gt=False)
            del image
            
            assert(im_index==gt_index), f"im_index: {im_index} - gt_index: {gt_index}"
            
            case_dict[tcase] = case_dict.get(tcase, []) + list(range(old_index, im_index))
            print(f"case dict: {old_index} to {im_index}")
            
            for k, v in class_dict.items():
                print(f"{k}:{len(v)}", end = "||")
            
            with open(case_dict_outpath, "w") as outfile: 
                json.dump(case_dict, outfile)
            with open(class_dict_outpath, "w") as outfile: 
                json.dump(class_dict, outfile)
        
        print(tcase, case_dict[tcase])
        print("="*40)
    
    case_dict['last_index'] = im_index
        
    with open(case_dict_outpath, "w") as outfile: 
        json.dump(case_dict, outfile)
            
    print(f"[Summary]: {im_index} patches in total")
    
    
# gt_8052.png