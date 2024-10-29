import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

def open_img(img_path):
    out = cv2.imread(img_path)[:, :, :3]
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.uint8)
    return out

def laplacian_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mask_img = cv2.convertScaleAbs(laplac)
    return mask_img

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
        masks.append(mask.sum())

    return masks

def write2file(text, target_file, mode='r'):
    with open(target_file, mode) as f:
        f.write(f"{text}\n")
        
def add2dict(dict_, key, value):
    dict_[key] = value
    return dict_
        
def postprocess_counts(class_counts, target_file, image_dict):
    class_counts = class_counts[1:] # skip_background
    class_percent = np.array(class_counts) / np.sum(np.array(class_counts))
    
    labels = ['background', 
            'viable',
            'necrosis',
            'fibrosis/hyalination',
            'hemorrhage/cystic-change',
            'inflammatory',
            'non-tumor']
        
    for i in range(6):
        # print(f"{labels[i+1]} - {round(class_percent[i], 4)}")
        write2file(f"{labels[i+1]} - {round(class_percent[i]*100, 1)}%", target_file, 'a')
        image_dict = add2dict(image_dict, labels[i+1], class_percent[i])
            
    huvos_ratio = 1 - class_counts[0] / (np.sum(class_counts[:5]) + 1e-9)
    
    if class_percent[-1] >= 0.99:
        huvos_ratio = None
        
    if huvos_ratio is not None:
        write2file(f'total_necrosis: {round(huvos_ratio*100, 1)}%', target_file, 'a')
    else:
        write2file('total_necrosis: N/A', target_file, 'a')
    write2file("--------------\n", target_file, 'a')
    
    return huvos_ratio, image_dict

def process_folder(label_folder, image_folder, outdir, target_file, case_dict):
    
    huvos_case = []
    label_names = [n for n in os.listdir(label_folder) if ('.jpg' in n or '.png' in n)]
    
    for label_name in tqdm(label_names, total=len(label_names)):
        
        if "x8" in label_name:
            image_name = label_name.split("-x8")[0] + '.png'
            upsample = True
        else:
            image_name = label_name.split("-labels")[0] + '.png'
        
        image_dict = {}
        
        write2file(image_name, target_file, 'a')
        label_path = os.path.join(label_folder, label_name)
        
        label = open_img(label_path)
        label = cv2.resize(label, None, fx=5e-2, fy=5e-2, interpolation=cv2.INTER_NEAREST)
        # label = label * image_mask + np.ones_like(label) * 255 * (1 - image_mask)
        
        masks = apply_threshold_mapping(label)
        plt.imsave(
            os.path.join(outdir, label_name), label.astype(np.uint8)
        )
        
        huvos_ratio, image_dict = postprocess_counts(masks, target_file, image_dict)
        if huvos_ratio is not None: 
            huvos_case.append(huvos_ratio)
        
        case_dict[image_name] = image_dict
        
    return np.mean(huvos_case), case_dict

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
        

if __name__ == '__main__':
    label_folder = '/workdir/radish/manhduong/labels'
    image_folder = '/workdir/radish/manhduong/images'
    outdir = "/home/manhduong/BoneTumor/RAW/REAL_WSIs/REAL_STATISTICS"
    os.makedirs(outdir, exist_ok=True)
    done_cases = []
    cases = ["Case_6", "Case_8"]
    
    metadatas = {}
    
    for case in os.listdir(label_folder):
        if case in done_cases: continue
        if case not in cases: continue
        
        print(f"Processing: {case}...")
        
        case_dict = {}
        
        case_outdir = os.path.join(outdir, case)
        os.makedirs(case_outdir, exist_ok=True)
        
        target_file = os.path.join(outdir, f"stats_{case.lower()}.txt")
        write2file(f"="*5 + f" {case} " + "="*5, target_file, 'w')
        
        case_label_folder = os.path.join(label_folder, case)
        case_image_folder = os.path.join(image_folder, case)
        
        if not (os.path.isdir(case_label_folder) and os.path.isdir(case_image_folder)): continue
        
        huvos_case, case_dict = process_folder(case_label_folder, case_image_folder, case_outdir, target_file, case_dict)
        huvos_case = round(huvos_case * 100, 1)
        write2file(f"Total Necrosis on case: {huvos_case}%", target_file, 'a')
        write2file(f"Huvos: {huvos_classify(huvos_case)}", target_file, 'a')
        
        metadatas[case] = case_dict
        
        # rewrite
        with open(os.path.join(outdir, 'gt_dict.json'), 'w') as f:
            json.dump(metadatas, f, indent=4)