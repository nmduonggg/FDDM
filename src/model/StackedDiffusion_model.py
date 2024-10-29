import os
import sys
sys.path.append('/home/user01/aiotlab/nmduong/BoneTumor/BBDM2')

import torch  
import torch.nn as nn  
import torch.nn.functional as F

import timm
import loralib as lora
# from huggingface_hub import hf_hub_download
import numpy as np
import matplotlib.pyplot as plt
import data.utils as data_utils

from model import UNI_lora_cls
from model.bbdm.BrownianBridge.LatentBrownianBridgeModel_PathologyContext import LatentBrownianBridgeModel_Pathology as LBBDM


def generate_color_boundaries(color_map, threshold=20):
    boundaries = []
    for color in color_map:
        lower_bound = [max(0, c - threshold) for c in color]
        upper_bound = [min(255, c + threshold) for c in color]
        boundaries.append({
            'color': color,
            'lower_bound': np.array(lower_bound),
            'upper_bound': np.array(upper_bound)
        })
    return boundaries

class StackedDiffusionModel(nn.Module):
    def __init__(self, option):
        super(StackedDiffusionModel, self).__init__()
        self.option=option
        self.phase1_classifier = UNI_lora_cls(option['network_G']['out_nc'])
        
        self.phase2_refiner = LBBDM(option['bbdm']['model'])
        self.patch_size = 256
        self.phase2_size = 256
        self.color_map = [
            [255, 255, 255],    # background
            [0, 128, 0],    # Viable tumor
            [255, 143, 204],    # Necrosis
            [255, 0, 0],    # Fibrosis/Hyalination
            [0, 0, 0],  # Hemorrhage/ Cystic change
            [165, 42, 42],  # Inflammatory
            [0, 0, 255]    # Non-tumor tissue
        ]
        self.boundaries = generate_color_boundaries(self.color_map, threshold=50)
        
    def forward(self, x, infer=True):
        # BxCxHxW
        batch_size = x.size(0)
        
        # run phase 1
        original_preds, num_h, num_w, h, w = self._generate_patch_seq(x) 
        out_ori = self._combine_tensor(
                        original_preds, num_h, num_w, h, w, self.patch_size, self.patch_size,
                        batch_size=batch_size, channel=7)   # B, H, W, C
        
        out_indices = torch.argmax(out_ori, dim=-1)   # BxHxW
        out = torch.zeros_like(out_ori.permute(0,3,1,2))    # Bx7xHxW
        
        out.scatter_(1, out_indices.unsqueeze(1), 1)    # Bx7xHxW 
        out = out.unsqueeze(2) * \
            torch.tensor(self.color_map).reshape(1, len(self.color_map), -1, 1, 1).to(out.device)   # Bx7x3xHxW
        out1 = torch.sum(out, dim=1)    # Bx3xHxW

        
        # out1 = self._combine_tensor(
        #                 patch_seq, num_h, num_w, h, w, self.patch_size, self.patch_size,
        #                 batch_size=batch_size, channel=3)   # B, H, W, C
        
        out_ori = F.interpolate(out_ori.permute(0,3,1,2), 
                                (self.phase2_size, self.phase2_size)).permute(0,2,3,1)
        
        # out1 = out1.permute(0, 3, 1, 2) / 255.  # original color 0,1
        out1 = out1 / 255.
        
        x = data_utils.denormalize_tensor(x)
        
        x_cond = F.interpolate(out1.to(x.device), (self.phase2_size, self.phase2_size))
        x_cont = F.interpolate(x, (self.phase2_size, self.phase2_size))
        
        x_cond = self._to_normal(x_cond)
        x_cont = self._to_normal(x_cont)
        
        x_cont = torch.cat([x_cond, x_cont], dim=1)
        
        out = self.phase2_refiner.sample_infer(x_cond, x_cont, clip_denoised=self.option['bbdm']['clip_denoised'])
        out = self._rm_normal(out)
        
        if infer:
            out = self.remap_color(out)
            out = out[0].permute(1,2,0)  # B, 3, H, W -> H, W, 3
            # np_out = out.cpu().numpy()
            # plt.imsave('./sample.png', np_out)
            out = self._convert_mapping_tensor(out).unsqueeze(0)    # hxwx7
            # out = torch.tensor(out).unsqueeze(0).to(x.device)
            
            out_ori = torch.softmax(out_ori, dim=-1)    # convert to prob vector
            # out_ori = torch.softmax(out_ori, dim=-1)
            out = out * 0.9 + out_ori
            # out_mask = torch.amax(out_ori, dim=-1, keepdim=True) > 0.5   # BxHxW
            
            # out = out * (~out_mask) + out_ori * out_mask
            # out = out
        
        return out
    
    # def _convert_mapping(self, image):
    #     # Create masks for pixels that are closer to green or pink
    #     # Initialize the output image with the original image
    #     tolerance = 85
        
    #     if np.max(image) <= 1:
    #         image = (image * 255).astype(np.uint8)
        
    #     output = np.zeros(list(image.shape[:2]) + [7])
    #     masks = []
    #     for idx, color in enumerate(self.color_map):
    #         color = np.array(color)
    #         mask = np.all(np.abs(image - color) < tolerance, axis=-1)
    #         mask = np.expand_dims(mask, axis=-1)    #hxwx1
    #         # output[mask] = color
    #         if mask.sum() > 0:
    #             class_mask = np.zeros_like(output)
    #             class_mask[:, :, idx] = 1.0
    #             output = output * (1-mask) + mask * class_mask
    #             # print(idx, mask.mean())

    #     return output
    

    def _convert_mapping_tensor(self, image):
        # Create masks for pixels that are closer to green or pink
        # Initialize the output image with the original image
        tolerance = 5
        
        # Nếu giá trị lớn nhất của tensor nhỏ hơn hoặc bằng 1, chuyển đổi nó sang kiểu uint8
        # if torch.max(image) <= 1:
        image = (image * 255).to(torch.uint8)
        
        # Tạo tensor đầu ra với kích thước (h, w, 7)
        output = torch.zeros(list(image.shape[:2]) + [7], dtype=torch.float32).to(image.device)
        
        masks = 0.
        for idx, color in enumerate(self.color_map):
            color = torch.tensor(color, dtype=torch.uint8).to(image.device)
            
            # Tạo mặt nạ bằng cách tìm các pixel có giá trị gần với màu cụ thể
            mask = torch.all(torch.abs(image - color) < tolerance, dim=-1).to(image.device)
            
            mask = mask.unsqueeze(-1)  # Thêm chiều mới để có kích thước (h, w, 1)
            
            # # Nếu mặt nạ có giá trị hợp lệ, tạo mặt nạ lớp tương ứng
            if mask.sum() > 0:
                class_mask = torch.zeros_like(output).to(image.device)
                class_mask[:, :, idx] = 1.0
            
                # Cập nhật output bằng cách sử dụng mặt nạ
                output = output * (~mask) + mask * class_mask
            # output += class_mask
        # print(output.max())
                
            # masks += mask.float().mean()
        # print(masks)

        return output
    
    def remap_color(self, image):
        # Define lower and upper bounds for blue-like pixels
        image = (image * 255).to(torch.uint8)
        new_image = image.clone()
        for color in self.boundaries:
        
            lb = torch.tensor(color['lower_bound'], dtype=torch.uint8).view(1, 3, 1, 1).to(image.device)
            ub = torch.tensor(color['upper_bound'], dtype=torch.uint8).view(1, 3, 1, 1).to(image.device)

            # Create a mask for blue-like pixels
            mask = ((image >= lb) & (image <= ub)).all(dim=1).to(image.device)
            if mask.sum() == 0: continue
            # Set all blue-like pixels to standard blue [0, 0, 255]
            new_image[:, :, mask.squeeze(0)] = torch.tensor(color['color'], dtype=torch.uint8).view(3, 1).to(image.device)
        
        return new_image / 255.
        
    def _to_normal(self, x):
        x = (x - 0.5) * 2.
        x = torch.clamp(x, -1, 1)
        return x
    
    def _rm_normal(self, x):
        x = x * 0.5 + 0.5
        x = torch.clamp(x, 0, 1.)
        return x

    def _generate_patch_seq(self, x):
        img_list, num_h, num_w, h, w = self._crop_tensor(x, self.patch_size, self.patch_size)
        
        outs = []
        img_tensor = torch.stack(img_list, dim=0)   # length_of_seq x B x CxHxW
        length, B, C, H, W = img_tensor.shape
        img_tensor = img_tensor.reshape(length*B, C, H, W).to(x.device)
        
        preds = self.phase1_classifier(img_tensor) 
        preds = preds.reshape(length, B, -1)    # length_of_seq x B x C
        
        preds = preds.reshape(list(preds.shape) + [1, 1]) * \
            torch.ones((list(preds.shape) + [self.patch_size, self.patch_size])).to(preds.device)   # lenghtxBxCxHxW
            
        # out_indices = torch.argmax(preds, dim=2)   # SxBxHxW
        # out = torch.zeros_like(preds)
        
        # out.scatter_(2, out_indices.unsqueeze(2), 1)    # SxBxCxHxW
        # out = out.unsqueeze(3) * \
        #     torch.tensor(self.color_map).reshape(1, 1, len(self.color_map), -1, 1, 1).to(out.device)   # SxBxCx3
        # out = torch.sum(out, dim=2)    # SxBx3xHxW
        
        
        return preds, num_h, num_w, h, w
    
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
                sr_subim = sr_subim.permute(0,2,3,1)    # BxCxHxW -> BxHxWxC
                
                sr_img[:, i*step: i*step+patch_size, j*step: j*step+patch_size,:] += sr_subim    # BxHxWxC * BxC
                index+=1

        for j in range(1,num_w):
            sr_img[:,j*step:j*step+(patch_size-step),:]/=2

        for i in range(1,num_h):
            sr_img[i*step:i*step+(patch_size-step),:,:]/=2
        return sr_img
        
    def load_state_dict(self, phase1_dict, phase2_dict, strict=True):
        self.phase1_classifier.load_state_dict(phase1_dict, strict=strict)
        self.phase2_refiner.load_state_dict(phase2_dict['model'], strict=strict)