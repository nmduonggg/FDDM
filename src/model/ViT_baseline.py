import os

import torch  
import torch.nn as nn  
import torch.nn.functional as F

import timm
import loralib as lora
from huggingface_hub import hf_hub_download

class ViT_baseline(nn.Module):
    def __init__(self, out_nc):
        super(ViT_baseline, self).__init__()
        
        # model = timm.create_model("hf-hub:MahmoodLab/uni", img_size=256,
        #                           pretrained=True, init_values=1e-5, dynamic_img_size=True)
        model = timm.create_model(
            "vit_large_patch16_224", img_size=256, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        
        self.tile_encoder = model
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, out_nc)
        )
        # self.apply_lora_to_vit(16, 32)

    def encode(self, x):
        # Forward pass through the ViT model with LoRA
        feature = self.tile_encoder(x)
        return feature
        
    def forward(self, x):
        bs, c, h, w = x.shape
        feature = self.tile_encoder(x)
        out = self.classifier(feature)
        return out
    
    def full_forward(self, x):
        bs, c, h, w = x.shape
        feature = self.tile_encoder(x)
        out = self.classifier(feature)
        return feature, out
    
    # def apply_lora_to_vit(self, lora_r, lora_alpha):
    #     """
    #     Apply LoRA to all the Linear layers in the Vision Transformer model.
    #     """
    #     # Step 1: Collect the names of layers to replace
    #     layers_to_replace = []
        
    #     for name, module in self.tile_encoder.named_modules():
    #         if isinstance(module, nn.Linear) :
    #             if 'qkv' in name or 'proj' in name:
    #                 # Collect layers for replacement (store name and module)
    #                 layers_to_replace.append((name, module))
        
    #     # Step 2: Replace the layers outside of the iteration
    #     for name, module in layers_to_replace:
    #         # Create the LoRA-augmented layer
    #         lora_layer = lora.Linear(module.in_features, module.out_features, r=lora_r, lora_alpha=lora_alpha)
    #         # Copy weights and bias
    #         lora_layer.weight.data = module.weight.data.clone()
    #         if module.bias is not None:
    #             lora_layer.bias.data = module.bias.data.clone()

    #         # Replace the layer in the model
    #         parent_name, layer_name = name.rsplit('.', 1)
    #         parent_module = dict(self.tile_encoder.named_modules())[parent_name]
    #         setattr(parent_module, layer_name, lora_layer)

    # # Additional helper to enable LoRA fine-tuning
    # def enable_lora_training(self):
    #     # Set LoRA layers to be trainable, freeze others
    #     for param in self.tile_encoder.parameters():
    #         param.requires_grad = False
        
    #     for name, param in self.tile_encoder.named_parameters():
    #         if "lora" in name:
    #             param.requires_grad = True

    #     # Enable gradients for the classifier head
    #     for param in self.classifier.parameters():
    #         param.requires_grad = True