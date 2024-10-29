import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from tqdm import tqdm

import options.options as option
from model import create_model
import utils.utils as utils

abspath = os.path.abspath(__file__)
import faulthandler; faulthandler.enable()
# Image.MAX_IMAGE_PIXELS = (2e40).__str__()


parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.')
parser.add_argument('-root', type=str, default=None, choices=['.'])
parser.add_argument('--labels_dir', type=str, required=True)
parser.add_argument('--images_dir', type=str, required=True)
parser.add_argument('--weight_path', type=str, required=True)
parser.add_argument('--old_labels_dir', type=str, required=True)
args = parser.parse_args()
opt = option.parse(args.opt, root=args.root)

opt = option.dict_to_nonedict(opt)

# Init
crop_sz = 256
step = 256
infer_size = 256
small_h = small_w = 16
ratio = int(crop_sz / small_h)
small_step = step // ratio
color_map = [
    [255, 255, 255],    # background
    [0, 128, 0],    # Viable tumor
    [255, 143, 204],    # Necrosis
    [255, 0, 0],    # Fibrosis/Hyalination
    [0, 0, 0],  # Hemorrhage/ Cystic change
    [165, 42, 42],  # Inflammatory
    [0, 0, 255]]    # Non-tumor tissue

def apply_threshold_mapping(image):
    # Create masks for pixels that are closer to green or pink
    # Initialize the output image with the original image
    tolerance = 50
    
    output = np.ones_like(image)*255 # 2D only
    masks = []
    for idx, color in enumerate(color_map):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        output[mask] = color
        # output[mask] = idx

    return output

def read_percent_from_color(image):
    tolerance = 20
    masks = []
    for idx, color in enumerate(color_map):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        masks.append(mask.sum())

    return masks

def open_img(path):
    img = cv2.imread(path)[:, :, :3].astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def prepare(infer_path):
    global crop_sz, step
    
    img = open_img(infer_path)
    
    return [img, utils.crop(img, crop_sz, step)]
    
def postprocess(**kwargs):
    return utils.combine(**kwargs)

def laplacian_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mask_img = cv2.convertScaleAbs(laplac)
    return mask_img

def index2color(idx, patch, color_map):
    
    # patch = F.interpolate(patch, (crop_sz, crop_sz))
    
    color = color_map[idx]
    color_tensor = torch.ones_like(patch) * torch.tensor(color).reshape(1, -1, 1, 1)
    color_np = color_tensor.squeeze(0).permute(1,2,0).numpy() / 255.0
    
    # idx += 1 # [0, 1, 2, 3]
    class_tensor = torch.ones_like(patch) * torch.tensor(idx).reshape(1, 1, 1, 1)
    class_np = class_tensor.squeeze(0).permute(1,2,0).numpy() 
    
    return color_np, idx

def alpha_blending(im1, im2, alpha):
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')
    out = cv2.addWeighted(im1, alpha , im2, 1-alpha, 0)
    return out

def infer(infer_path, label_path, old_label_path):
    global color_map
    infer_name = os.path.basename(infer_path)
    
    print("Image path: ", infer_path)
    print("Original label path: ", label_path)
    print("Old prediction: ", old_label_path)
    
    with open(os.path.join(working_dir, "result.txt"), "a") as f:
        f.write(f"{infer_name}\n")
    
    fx = 5e-2
    fy = 5e-2
    
    prediction = open_img(old_label_path)
    
    print("start read label")
    label_mask = open_img(label_path)
    print("reading done")
    label_mask = cv2.resize(label_mask, (prediction.shape[1], prediction.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    plt.imsave(os.path.join(working_dir, 'label_mask.png'), label_mask)
    label_mask = np.any(np.abs(label_mask - np.array([255, 255, 255])) > 0, axis=-1).astype(int)
    
    label_mask = np.expand_dims(label_mask, axis=-1)
    
    print("create label mask done")
    
    prediction = prediction * label_mask + np.ones_like(prediction) * 255 * (1 - label_mask)
    class_counts = read_percent_from_color(prediction)
    
    plt.imsave(os.path.join(working_dir, f"{infer_name.split('.')[0]}_pred.png"), prediction.astype('uint8')) 
    
    img = open_img(infer_path)
    plt.imsave(os.path.join(working_dir, f"{infer_name.split('.')[0]}.png"), img.astype('uint8')) 
    img = cv2.resize(img, (prediction.shape[1], prediction.shape[0]))
    # label_mask = cv2.resize(label_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # cut to align img and prediction
    img = img * label_mask + np.ones_like(img) * 255 * (1 - label_mask)
    
    blend_im = alpha_blending(img, prediction, 0.6)
    
    del img
    del prediction  # free mem
    
    class_counts = class_counts[1:] # skip_background
    class_percent = np.array(class_counts) / np.sum(np.array(class_counts))
    print(class_percent)
    
    labels = ['background', 'viable', 'necrosis', 'fibrosis/hyalination', 'hemorrhage/cystic-change', 'inflammatory', 'non-tumor']

    plt.imsave(os.path.join(working_dir, f"{infer_name.split('.')[0]}_blend.png"),
               blend_im.astype(np.uint8))
    print("Save prediction done")
        
    print("[RESULT]")
    for i in range(6):
        # if i==0: continue
        # i += 1
        print(f"{labels[i+1]} - {round(class_percent[i], 4)}")
        with open(os.path.join(working_dir, "result.txt"), "a") as f:
            f.write(f"{labels[i+1]} - {(round(class_percent[i], 1)*100):.1f}%\n")
            
    huvos_ratio = 1 - class_counts[0] / (np.sum(class_counts[:5]) + 1e-9) 
    
    if class_percent[-1] >= 0.99:
        huvos_ratio = None
    
    with open(os.path.join(working_dir, "result.txt"), "a") as f:
        if huvos_ratio is not None:
            f.write(f'total_necrosis: {(round(huvos_ratio, 1)*100):.1f}% \n')
        else:
            f.write('total_necrosis: N/A \n')
        f.write(f"-------------\n")
        
    print("-"*20)
    
    return class_counts, huvos_ratio

def huvos_classify(huvos_ratio):
    labels = ["I", "II", "III", "IV"]
    index = 0
    if huvos_ratio < 50:
        index = 0
    elif 50 <= huvos_ratio < 90:
        index = 1
    elif 90 <= huvos_ratio < 100:
        index = 2
    else:
        index = 3
    return labels[int(index)]


if __name__=='__main__':
    
    
    done_cases = [f"Case_{n}" for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    
    for case in os.listdir(args.labels_dir):
        if case not in done_cases: continue
        # if case != 'Case_1': continue
        
        label_dir = os.path.join(args.labels_dir, case)
        image_dir = os.path.join(args.old_labels_dir, case) # Old visualizations
        old_label_dir = os.path.join(args.old_labels_dir, case)
        
        # initialize working dir for each case
        working_dir = os.path.join('.', 'infer', 'smooth_fixed', opt['name'], case)
        os.makedirs(working_dir, exist_ok=True)
        print("Working dir: ", working_dir)
    
        case_patch_counts = [0 for _ in range(len(color_map)-1)]
        with open(os.path.join(working_dir, "result.txt"), "w") as f:
            f.write(f"======={case}=======\n")
            
        huvos_case = []
        
        label_names = [n for n in os.listdir(label_dir) if ('.jpg' in n or '.png' in n)]

        for label_name in label_names:
            
            # start
            
            if "x8" in label_name:
                image_name = label_name.split("-x8")[0] + '.png'
                upsample = True
            else:
                image_name = label_name.split("-labels")[0] + '.png'
                
            
            infer_path = os.path.join(old_label_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
            
            old_label_name = image_name.split(".png")[0] + '_pred.png'
            old_label_path = os.path.join(old_label_dir, old_label_name)
                
            print("Process: ", infer_path)
            slide_patch_counts, huvos_ratio = infer(infer_path, label_path, old_label_path)
            if huvos_ratio is not None:
                huvos_case.append(huvos_ratio)
            
            for i, class_count in enumerate(slide_patch_counts):
                case_patch_counts[i] += slide_patch_counts[i]
                
        case_patch_percents = np.array(case_patch_counts) / np.sum(np.array(case_patch_counts))
        
        huvos_case = np.mean(huvos_case)
        huvos_case = round(huvos_case*100, 1)
        
        with open(os.path.join(working_dir, "result.txt"), "a") as f:
            f.write(f"Total Necrosis on case: {huvos_case:.1f}%\n")
            f.write(f"Huvos: {huvos_classify(huvos_case)}")
        
            
            
            
            
            