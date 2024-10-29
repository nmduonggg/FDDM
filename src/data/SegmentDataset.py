import numpy as np
import pandas as pd
import os
import pickle
import random
import cv2
import torch
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

from data import utils
import albumentations as A

def apply_threshold_mapping(image, target_colors, tolerance):
    # Create masks for pixels that are closer to green or pink
    # Initialize the output image with the original image
    output = np.zeros_like(image)[:, :, 0] # 2D only
    masks = []
    for idx, color in enumerate(target_colors):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        # output[mask] = color
        output[mask] = idx

    return output

def get_blank_mask(image, target_colors, tolerance=5):
    # Create masks for pixels that are closer to green or pink
    # Initialize the output image with the original image
    
    color = [255, 255, 255]
    color = np.array(color)
    mask = np.all(np.abs(image - color) < tolerance, axis=-1)

    return mask

class SegmentDataset(Dataset):
    def __init__(self, opt):
        
        self.image_dir = opt['image_dir']
        self.label_dir = opt['label_dir']
        # self.size = (opt['height'], opt['width']) if opt['height'] is not None else None
        self.opt = opt
        self.n_classes = 7
        
        self.root_folder = opt['root']
        
        with open(os.path.join(opt['root'], 'dataset_split.json'), 'r') as f:
            self.indices = json.load(f)[opt['type']]
            
        with open(os.path.join(opt['root'], 'metadata_reindex.json'), 'r') as f:
            self.data_list = json.load(f)
        
        print("Number of classes: ", self.n_classes)
        # with open(opt['label_map'], 'r') as f:
        #     self.indices = json.load(f)[opt['type']]
            
        self.target_colors = [
            [255, 255, 255],
            [0, 128, 0],
            [255, 143, 204],
            [255, 0, 0],
            [0, 0, 0],
            [165, 42, 42],
            [0, 0, 255]]
        self.tolerance = 50
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        

        self.augmentation = A.Compose([
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),]
        )
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        
        # item_idx = self.indices[index]
        item_infor = self.data_list[self.indices[index]]
        img_path = os.path.join(self.image_dir, f"{item_infor['crop_index']}.png")
        
        x = Image.open(os.path.join(img_path)).convert("RGB")
        
        y = cv2.imread(os.path.join(self.label_dir, f"{item_infor['crop_index']}.png"))[:, :, :3]
        y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
        y = cv2.resize(y, (256, 256), interpolation=cv2.INTER_NEAREST)
        y = apply_threshold_mapping(y, self.target_colors, self.tolerance)
        
        x = self.transform(x).float()
        # print(x.shape)
        y = torch.tensor(y).long()
        
        return x, y
