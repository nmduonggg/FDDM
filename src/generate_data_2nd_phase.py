import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
# import sys
# sys.setrecursionlimit(1500)


import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import options.options as option
from model import create_model
import utils.utils as utils
# import data.utils as data_utils

from huggingface_hub import login

abspath = os.path.abspath(__file__)


parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.')
parser.add_argument('-root', type=str, default=None, choices=['.'])

parser.add_argument('--outdir', type=str, required=True,)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--weight_path', type=str, required=True)

#-------------------------------------#
args = parser.parse_args()
opt = option.parse(args.opt, root=args.root)

opt = option.dict_to_nonedict(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % opt['gpu_ids'][0]
device = torch.device('cuda:0' if opt['gpu_ids'] is not None else 'cpu')
# device = torch.device('cpu')

# HF Login to get pretrained weight
# login(opt['token'])
# import subprocess
# subprocess.run(["huggingface-cli", "login", "--token", opt['token']])
# login(opt['token'])
    
model = create_model(opt)

if args.weight_path != '':
    state_dict = torch.load(args.weight_path, map_location='cpu')
    _strict = True

    model.load_state_dict(state_dict, strict=_strict)  
    print("[INFO] Load weight from:", args.weight_path)
else:
    print("No pretrained weight found")
    
# Init
crop_sz = 256
step = 256

color_map = [
    [255, 255, 255],    # background
    [0, 128, 0],    # Viable tumor
    [255, 143, 204],    # Necrosis
    [255, 0, 0],    # Fibrosis/Hyalination
    [0, 0, 0],  # Hemorrhage/ Cystic change
    [165, 42, 42],  # Inflammatory
    [0, 0, 255]]    # Non-tumor tissue

def write2file(metadata, metadata_file, mode='a'):
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
    with open(metadata_file, mode) as f:
        json.dump(metadata, f, indent=4, cls=NpEncoder)

def apply_threshold_mapping(image, target_colors, tolerance=50):
    
    masks = []
    for idx, color in enumerate(target_colors):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        masks.append(mask.mean())
    return np.argmax(np.array(masks))

def fix_label_image(im, label):
    im_mask = np.any(np.abs(im - np.array([255, 255, 255])) > 10, axis=-1)
    im_mask = np.expand_dims(im_mask, axis=-1)
    label = label * im_mask + np.ones_like(label) * 255 * (1 - im_mask)
    
    label_mask = np.any(np.abs(label - np.array([255, 255, 255])) > 10, axis=-1)
    label_mask = np.expand_dims(label_mask, axis=-1)
    im = im * label_mask + np.ones_like(im) * 255 * (1 - label_mask)
    
    return im, label

def prepare(infer_path):
    global crop_sz, step
    
    img = cv2.imread(infer_path)[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img, utils.crop(img, crop_sz, step)


def infer(image_filepath, label_filepath, size=256):
    global color_map
    
    model.to(device)
    model.eval()
    
    img, preprocess_elems = prepare(image_filepath)
    image_patches, num_h, num_w, h, w = preprocess_elems
    label, preprocess_elems = prepare(label_filepath)
    label_patches = preprocess_elems[0]
    assert(len(image_patches)==len(label_patches))
    
    # img = cv2.resize(img, (size, size)).astype('uint8')
    # label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST).astype('uint8')
    
    preds_list = []
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    for im_patch, gt_patch in zip(image_patches, label_patches):
        
        assert(im_patch.size == gt_patch.size)
        # im_patch, gt_patch = fix_label_image(im_patch, gt_patch)
        
        bg = np.ones((crop_sz, crop_sz, 3), 'float32') * 255
        r, c, _ = im_patch.shape
        bg[:r, :c, :] = im_patch
        im_patch = bg.astype(np.uint8)
        
        im = transform(im_patch).float().unsqueeze(0)   # Bx3xHW
        # b, _, h, w = im.shape
        
        im = im.to(device)
        with torch.no_grad():
            pred = model(im)
        
        # pred = torch.argmax(pred, dim=-1).cpu().squeeze(0).item()
        # pred_im = np.ones_like(im_patch) * np.array(color_map[pred]).reshape(1,1,-1)
        pred_im = torch.ones((size, size, 1)) * pred.cpu().reshape(1, 1, -1)    # -> HxWxC
        preds_list.append(pred_im.numpy())
    # print(num_h, num_w, h, w)
    
    pred = utils.combine(preds_list, num_h, num_w, h, w, size, step, len(color_map))
    # pred = cv2.resize(pred, (size, size), interpolation=cv2.INTER_NEAREST).astype('uint8')
    # pred = cv2.resize(pred, (size, size))
    
    return pred, label, img


if __name__=='__main__':
    
    input_folder = os.path.join(args.outdir, 'false_pred')
    label_folder = os.path.join(args.outdir, 'label')
    img_folder = os.path.join(args.outdir, 'images')
    metadata_outfile = os.path.join(args.outdir, 'metadata.json')
    
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)
    
    if os.path.isfile(metadata_outfile):
        with open(metadata_outfile, 'r') as f:
            original_data_list = json.load(f)
        print("Load original metadat from: ", metadata_outfile)
    else: 
        original_data_list = list()
        print("Initialize new data list")
        
    print("Save to: ", args.outdir)
    os.makedirs(args.outdir, exist_ok=True)
        
    #----------------------------------------------# processing
    
    done_cases = [f"Case_{i}" for i in []]
    # process_cases = ["Case_9"]
    
    for case_name in os.listdir(args.data_dir):
        if "Case" not in case_name: continue
        if case_name in done_cases: continue
        # if case_name not in process_cases: continue
        
        print(f"==========={case_name}===========")
        case_dir = os.path.join(args.data_dir, case_name)
    
        metadata_file = os.path.join(case_dir, 'metadata.json')
        with open(metadata_file, 'r') as f:
            data_list = json.load(f)
        
        # original_data_list += data_list
        cnt = 0
        print("Start from crop_index:", data_list[0]['crop_index'])
        for metadata in tqdm(data_list, total=len(data_list)):
            crop_image_name = metadata['croped_image_filename']
            crop_label_name = metadata['croped_label_filename']
            crop_index = metadata['crop_index']
                
            # start
            image_filepath = os.path.join(case_dir, 'images', crop_image_name)
            label_filepath = os.path.join(case_dir, 'labels', crop_label_name)                
            
            pred_image, label_image, img_image = infer(image_filepath, label_filepath)
            
            # plt.imsave(os.path.join(input_folder, f"{crop_index}.png"), pred_image)
            np.save(os.path.join(input_folder, f"{crop_index}.npy"), pred_image)    # hxwx7
            plt.imsave(os.path.join(label_folder, f"{crop_index}.png"), label_image)
            plt.imsave(os.path.join(img_folder, f"{crop_index}.png"), img_image)
            
            metadata['case'] = metadata.get('case', case_name) 
            original_data_list.append(metadata)
            
            if cnt % 1000 == 0:
                write2file(original_data_list, metadata_outfile, 'w')
                print("[INFO] Write metadata")
            cnt += 1
            
        print(f"{case_name} - {cnt} images")
                
                