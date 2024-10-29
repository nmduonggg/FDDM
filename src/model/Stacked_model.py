import os

import torch  
import torch.nn as nn  
import torch.nn.functional as F

import timm
import loralib as lora
from huggingface_hub import hf_hub_download
import numpy as np

from model import TransformerReorder, UNI_lora_cls

class StackedModel(nn.Module):
    def __init__(self, out_nc):
        super(StackedModel, self).__init__()
        
        self.phase1_classifier = UNI_lora_cls(out_nc)
        self.phase2_reorder = TransformerReorder()
        self.patch_size = 256
        
    def forward(self, x):
        # BxCxHxW
        batch_size = x.size(0)
        patch_seq, num_h, num_w, h, w = self._generate_patch_seq(x) # -> BxSx1025
        mask = torch.zeros(len(patch_seq)).unsqueeze(0).bool().to(x.device) # no need to mask due to no padding
        
        input_seq = torch.stack(patch_seq, dim=1)
        patch_classification = self.phase2_reorder(input_seq, mask) # -> BxTxC
        output_seq = [patch_classification[:, i, :] for i in range(len(patch_seq))]
        
        out = self._combine_tensor(
                        output_seq, num_h, num_w, h, w, self.patch_size, self.patch_size,
                        batch_size=batch_size, channel=patch_classification.shape[-1])
        
        return out
    
    def _generate_patch_seq(self, x):
        img_list, num_h, num_w, h, w = self._crop_tensor(x, self.patch_size, self.patch_size)
        
        outs = []
        for img in img_list:
            feat, pred = self.phase1_classifier.full_forward(img)    # Bx1024 and B
            pred = torch.argmax(pred, dim=-1, keepdims=True)
            out = torch.cat([feat, pred], dim=-1)   # Bx1025
            outs.append(out)
            
        # outs = [Bx1025] x S
        
        return outs, num_h, num_w, h, w
    
    def _crop_tensor(self, img, crop_sz, step):
        # img: BxCxHxW
        b, c, h, w = img.shape
        h_space = np.arange(0, h - crop_sz + 1, step)
        w_space = np.arange(0, w - crop_sz + 1, step)
        index = 0
        num_h = 0
        lr_list=[]
        for x in h_space:
            num_h += 1
            num_w = 0
            for y in w_space:
                num_w += 1
                index += 1
                crop_img = img[:, :, x:x + crop_sz, y:y + crop_sz]
                lr_list.append(crop_img)
        h=x + crop_sz
        w=y + crop_sz
        return lr_list, num_h, num_w, h, w
    
    def _combine_tensor(self, sr_list, num_h, num_w, h, w, patch_size, step, batch_size, channel=3):
        index=0
        device = sr_list[0].device
        
        sr_img = torch.zeros((batch_size, h, w, channel)).to(device)
        for i in range(num_h):
            for j in range(num_w):
                sr_subim = sr_list[index]
                
                sr_img[:, i*step: i*step+patch_size, j*step: j*step+patch_size,:] += \
                    torch.ones((batch_size, step, step, channel)).to(sr_subim.device) * sr_subim
                index+=1

        for j in range(1,num_w):
            sr_img[:,j*step:j*step+(patch_size-step),:]/=2

        for i in range(1,num_h):
            sr_img[i*step:i*step+(patch_size-step),:,:]/=2
        return sr_img
        
    def load_state_dict(self, phase1_dict, phase2_dict, strict=True):
        self.phase1_classifier.load_state_dict(phase1_dict, strict=strict)
        self.phase2_reorder.load_state_dict(phase2_dict, strict=strict)
        
        return
            