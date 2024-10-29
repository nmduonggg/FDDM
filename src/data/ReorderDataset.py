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

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

class ReorderDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.input_dir = os.path.join(opt['data_dir'], 'data_x')
        self.label_dir = os.path.join(opt['data_dir'], 'data_y')
        self.metadata = read_json(
            os.path.join(opt['data_dir'], 'metadata.json'))
        self.indices = read_json(
            os.path.join(opt['data_dir'], 'dataset_split.json'))[opt['type']]
        self.max_length = opt['max_length']
        self.PAD = 7
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        
        item_idx = self.indices[index]
        item_infor = self.metadata[item_idx]
        
        name = f"{item_infor['crop_index']}.npy"
        x_path = os.path.join(self.input_dir, name)
        y_path = os.path.join(self.label_dir, name)
        
        x = torch.tensor(np.load(x_path)).float()   # Sx1025
        y = torch.tensor(np.load(y_path)).long()    # Sx1
        
        x, _ = self._pad_sequence(x)
        y, skip_mask = self._pad_sequence(y)
        
        x = x.float()
        y = y.squeeze(-1).long()
        
        return x, y, skip_mask
    
    def _pad_sequence(self, x):
        seq_len = x.size(0)
        padded_x = torch.zeros((self.max_length, x.size(1)))
        padded_x[:seq_len] = x
        padded_x[seq_len:, -1] = self.PAD
        
        skip_mask = torch.ones(self.max_length)
        skip_mask[:seq_len] = 0
        
        return padded_x, skip_mask.unsqueeze(0)