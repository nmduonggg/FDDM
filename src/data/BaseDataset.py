import numpy as np
import pandas as pd
import os
import pickle
import random
import cv2
import torch
from torch.utils.data import Dataset

from data import utils

class BaseDataset(Dataset):
    def __init__(self, opt):
        
        self.image_dir = opt['image_dir']
        self.label_dict = pd.read_csv(opt['label_map'])
        self.size = (opt['height'], opt['width']) if opt['height'] is not None else None
        self.opt = opt
        self.n_classes = len(list(set(self.label_dict['label'])))
        
        print("Number of classes: ", self.n_classes)
        
    def __len__(self):
        return self.label_dict.shape[0]
    
    def __getitem__(self, index):
        
        item_infor = self.label_dict.iloc[index]
        x = cv2.imread(os.path.join(self.image_dir, item_infor['name'] + '.jpg'))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        x = x / 255.0
        # random augment
        if self.opt['augment']:
            x = utils.augment(x, self.opt['use_flip'], self.opt['use_root'])
        
        y = item_infor['label']
        
        x = utils.normalize_np(x)
        x = torch.tensor(x).permute(2,0,1)
        x = utils.imresize(x.unsqueeze(0), self.size).squeeze(0)
        
        x = x.float()
        y = torch.tensor(y).long() 
        
        return x, y
