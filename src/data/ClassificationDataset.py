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
    
    masks = []
    for idx, color in enumerate(target_colors):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        # output[mask] = color
        # output[mask] = idx
        masks.append(mask.mean())
    return np.argmax(np.array(masks))

class ClassificationDataset(Dataset):
    def __init__(self, opt):
        
        self.image_dir = opt['image_dir']
        self.label_dir = opt['label_dir']
        # self.size = (opt['height'], opt['width']) if opt['height'] is not None else None
        self.opt = opt
        self.n_classes = 7
        
        print("Number of classes: ", self.n_classes)
        with open(opt['label_map'], 'r') as f:
            self.indices = json.load(f)[opt['type']]
            
        self.target_colors = [
            [255, 255, 255],    # background
            [0, 128, 0],    # Viable tumor
            [255, 143, 204],    # Necrosis
            [255, 0, 0],    # Fibrosis/Hyalination
            [0, 0, 0],  # Hemorrhage/ Cystic change
            [165, 42, 42],  # Inflammatory
            [0, 0, 255]]    # Non-tumor tissue
        self.tolerance = 50
        
        self.transform = transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.ToTensor(),
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
        
        item_idx = self.indices[index]
        # x = cv2.imread(os.path.join(self.image_dir, f"patch_{item_idx}.png"))
        # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = np.array(
            Image.open(os.path.join(self.image_dir, f"patch_{item_idx}.png")).convert("RGB"))
        
        y = cv2.imread(os.path.join(self.label_dir, f"gt_{item_idx}.png"))[:, :, :3]
        y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
        y = apply_threshold_mapping(y, self.target_colors, self.tolerance)
        
        # if abs(np.mean(x) - 255) < 20:
        #     y = 0
        
        if self.opt['augment']:
            # augmented = self.augmentation(image=x, mask=y)
            # x = augmented['image']
            # y = augmented['mask']
            
            # p = np.random.random()
            # if p > 0.8:
            #     x = np.ones_like(x) * 255
            #     y = np.ones_like(y) * 255
            # if int(y) not in [0, 6]:
            x = self.augmentation(image=x)['image']
        
        x = self.transform(x).float()
        y = torch.tensor(y).long() 
        
        return x, y
