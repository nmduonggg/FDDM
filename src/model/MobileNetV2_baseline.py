import os

import torch  
import torch.nn as nn  
import torch.nn.functional as F

import timm
import loralib as lora
from huggingface_hub import hf_hub_download

class MobileNetV2_baseline(nn.Module):
    def __init__(self, out_nc):
        super(MobileNetV2_baseline, self).__init__()
        
        # model = timm.create_model("hf-hub:MahmoodLab/uni", img_size=256,
        #                           pretrained=True, init_values=1e-5, dynamic_img_size=True)
        model = timm.create_model(
            "mobilenetv2_100", img_size=256, pretrained=True, num_classes=out_nc)
        
        self.tile_encoder = model

    def encode(self, x):
        # Forward pass through the ViT model with LoRA
        feature = self.tile_encoder(x)
        return feature
        
    def forward(self, x):
        bs, c, h, w = x.shape
        out = self.tile_encoder(x)
        return out
    