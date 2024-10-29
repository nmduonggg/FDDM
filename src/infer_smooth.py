import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import json

from torchvision import transforms
import options.options as option
from model import create_model
import utils.utils as utils

from huggingface_hub import login

abspath = os.path.abspath(__file__)
# Image.MAX_IMAGE_PIXELS = (2e40).__str__()


parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.')
parser.add_argument('-root', type=str, default=None, choices=['.'])
parser.add_argument('--labels_dir', type=str, required=True)
parser.add_argument('--images_dir', type=str, required=True)
parser.add_argument('--weight_path', type=str, required=True)
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--crop_sz', type=int, default=256)
parser.add_argument('--step', type=int, default=240)
parser.add_argument('--small_sz', type=int, default=16)
args = parser.parse_args()
opt = option.parse(args.opt, root=args.root)

opt = option.dict_to_nonedict(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % opt['gpu_ids'][0]
device = torch.device('cuda:0' if opt['gpu_ids'] is not None else 'cpu')

# HF Login to get pretrained weight
# login(opt['token'])
    
model = create_model(opt)

# fix and load weight
# state_dict = torch.load(opt['path']['pretrain_model'], map_location='cpu')

if args.weight_path != '':
    state_dict = torch.load(args.weight_path, map_location='cpu')
    # current_dict = model.state_dict()
    # new_state_dict={k:v if v.size()==current_dict[k].size()  else  current_dict[k] for k,v in zip(current_dict.keys(), state_dict.values())}    # fix the size of checkpoint state dict
    _strict = True

    model.load_state_dict(state_dict, strict=_strict)  
    print("[INFO] Load weight from:", args.weight_path)
else:
    print("No pretrained weight found")
    
# Init
crop_sz = args.crop_sz
step = args.step
infer_size = crop_sz
small_h = small_w = args.small_sz
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

def write2file(text, target_file, mode='r'):
    with open(target_file, mode) as f:
        f.write(f"{text}\n")

def add2dict(dict_, key, value):
    dict_[key] = value
    return dict_

def read_percent_from_color(image):
    tolerance = 50
    masks = []
    for idx, color in enumerate(color_map):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        masks.append(mask.sum())

    return masks

def open_img(image_path):
    img = cv2.imread(image_path)[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # pad_h = int(h % step)
    # pad_w = int(w % step)
    
    pad_h = (step - h % step) if h % step != 0 else 0
    pad_w = (step - w % step) if w % step != 0 else 0
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0,0)), mode='constant', constant_values=255)
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

def get_nonwhite_mask(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(grayscale, 250, 255, cv2.THRESH_BINARY)
    mask = (thresholded != 255).astype(int)
    
    return mask

def alpha_blending(im1, im2, alpha):
    out = cv2.addWeighted(im1, alpha , im2, 1-alpha, 0)
    return out

def infer(infer_path, label_path, target_file, outdir, image_dict):
    
    global color_map
    infer_name = os.path.basename(infer_path)
    fx = 5e-2
    fy = 5e-2
    
    model.to(device)
    model.eval()
    
    img, preprocess_elems = prepare(infer_path)
    patches_list, num_h, num_w, h, w = preprocess_elems
    kwargs = {
        'sr_list': patches_list,
        'num_h': num_h,
        'num_w': num_w,
        'h': h, 'w': w,
        'patch_size': crop_sz, 
        'step': step
    }
    # img = postprocess(**kwargs)
    img = img[:h,:w, :]
    img = cv2.resize(img, None, fx=fx, fy=fy)
    print(img.max())
    plt.imsave(os.path.join(outdir, infer_name), img) 
    print("Save original image done") 
    
    preds_list, class_list = [], []
    class_counts = [0 for _ in range(7)]
    for i, patch in tqdm(enumerate(patches_list), total=len(patches_list)):
        bg = np.ones((crop_sz, crop_sz, 3), 'float32') * 255
        r, c, _ = patch.shape
        bg[:r, :c, :] = patch
        patch = bg.astype(np.uint8)
        edge_score = laplacian_score(patch).mean()
        
        # normalize
        transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        im = transform(patch).float().unsqueeze(0)
        
        if edge_score <= 1: # filter background
            pred_im = np.zeros((small_h, small_w, 7))
            pred_im[:, :, 0] = 1e-9
            preds_list.append(pred_im)   # skip
            continue
        
        im = im.to(device)
        with torch.no_grad():
            pred = model(im)
        pred = torch.softmax(pred, dim=-1).cpu().squeeze(0).numpy()
        pred = np.ones((small_h, small_w, 7)) * pred
        preds_list.append(pred)
    
    kwargs['sr_list'] = preds_list
    kwargs['channel'] = 7
    kwargs['step'] = int(small_step)
    kwargs['patch_size'] = small_h
    kwargs['h'] = int(h / ratio)
    kwargs['w'] = int(w / ratio)
    
    prediction = postprocess(**kwargs)  # hxwx7
    prediction = np.expand_dims(np.argmax(prediction, axis=-1), axis=2)
    output = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype='uint8')
    
    for i in range(len(color_map)):
        color = np.array(color_map[i]).astype(np.uint8)
        mask = np.all(np.abs(prediction-i) < 1e-9, axis=-1)
        print(mask.sum())
        output[mask] = color
        
    prediction = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # img_mask = np.expand_dims(get_nonwhite_mask(img), axis=-1)
    
    label = cv2.resize(open_img(label_path), (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
    plt.imsave(os.path.join(outdir, f"{infer_name.split('.')[0]}_label.png"), label) 
    
    label_mask = np.expand_dims(get_nonwhite_mask(label), axis=-1)
        
    prediction = prediction * label_mask + np.ones_like(prediction)*255*(1-label_mask)
    prediction = prediction.astype(np.uint8)
    
    class_counts = read_percent_from_color(prediction)
    
    blend_im = alpha_blending(
        img, prediction, 0.6)
    
    plt.imsave(os.path.join(outdir, f"{infer_name.split('.')[0]}_pred.png"), prediction) 
    
    del prediction  # free mem
    
    class_counts = class_counts[1:] # skip_background
    class_percent = np.array(class_counts) / (np.sum(np.array(class_counts)) + 1e-9)
    print(class_percent)
    
    labels = ['background', 'viable', 'necrosis', 'fibrosis/hyalination', 'hemorrhage/cystic-change', 'inflammatory', 'non-tumor']

    plt.imsave(os.path.join(outdir, f"{infer_name.split('.')[0]}_blend.png"), blend_im.astype(np.uint8))
    print("Save prediction done")
    for i in range(6):
        write2file(f"{labels[i+1]} - {round(class_percent[i]*100, 1)}%", target_file, 'a')
        image_dict = add2dict(image_dict, labels[i+1], class_percent[i])
            
    huvos_ratio = 1 - class_counts[0] / (np.sum(class_counts[:5]) + 1e-9) 
    
    if class_percent[-1] >= 0.99:
        huvos_ratio = None
        
    if huvos_ratio is not None:
        write2file(f'total_necrosis: {round(huvos_ratio*100, 1)}%', target_file, 'a')
    else:
        write2file('total_necrosis: N/A', target_file, 'a')
    write2file("--------------", target_file, 'a')
    
    return huvos_ratio, image_dict

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

def process_folder(label_folder, image_folder, outdir, target_file, case_dict):
    huvos_case = []
    label_names = [n for n in os.listdir(label_folder) if ('.jpg' in n or '.png' in n)]
    for label_name in label_names:
        # if 'S7' not in label_name: continue
        if "x8" in label_name:
            image_name = label_name.split("-x8")[0] + '.png'
            upsample = True
        else:
            image_name = label_name.split("-labels")[0] + '.png'
        
        infer_path = os.path.join(image_folder, image_name)
        label_path = os.path.join(label_folder, label_name)
            
        image_dict = {}
        write2file(image_name, target_file, 'a')
        
        huvos_ratio, image_dict = infer(infer_path, label_path,
                                        target_file, outdir, image_dict)
        if huvos_ratio is not None:
            huvos_case.append(huvos_ratio)
            
        case_dict[image_name] = image_dict
            
    return np.mean(huvos_case), case_dict


if __name__=='__main__':
    done_cases = [f"Case_{n}" for n in [1, 3, 4, 9, 10]]
    # cases = [f"Case_{n}" for n in [6, 8]]
    cases = ["Case_6", "Case_8"]
    metadatas = {}
    
    outdir = args.outdir
    image_dir = args.images_dir
    label_dir = args.labels_dir
    print("Saving to dir:", outdir)
    
    pred_dict_path = os.path.join(outdir, 'pred_dict.json')
    if os.path.isfile(pred_dict_path):
        with open(pred_dict_path, 'r') as f:
            metadatas = json.load(f)
            print("Load metadatas from:", pred_dict_path)
    
    for case in os.listdir(args.labels_dir):
        if case in done_cases: continue
        if case not in cases: continue
        
        print("="*5 + f" {case} " + "="*5)
        
        case_label_folder = os.path.join(label_dir, case)
        case_image_folder = os.path.join(image_dir, case)
        
        case_dict = {}
        
        # initialize working dir for each case
        case_outdir = os.path.join(outdir, opt['name'], case)
        os.makedirs(case_outdir, exist_ok=True)
        
        target_file = os.path.join(outdir, f"stats_{case.lower()}.txt")
        write2file(f"="*5 + f" {case} " + "="*5, target_file, 'w')
        
        huvos_case, case_dict = process_folder(case_label_folder, case_image_folder, case_outdir, target_file, case_dict)
        huvos_case = round(huvos_case * 100, 1)
        write2file(f"Total Necrosis on case: {huvos_case}%", target_file, 'a')
        write2file(f"Huvos: {huvos_classify(huvos_case)}", target_file, 'a')
        
        metadatas[case] = case_dict
        with open(os.path.join(outdir, 'pred_dict.json'), 'w') as f:
            json.dump(metadatas, f, indent=4)
        
            
        
        
            
            
            
            
            