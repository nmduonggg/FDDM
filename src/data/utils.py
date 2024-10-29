import random
import numpy as np
import torch.nn.functional as F
import torch

def augment(img, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img
    
    aug = _augment(img)
    
    return aug

def add_noise(image, noise_typ='gauss'):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**2
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.001
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out

def imresize(im, size):
    return F.interpolate(im, size=size, mode='bicubic')

def normalize_np(im, rgb_mean=(0.485, 0.456, 0.406), rgb_std=(0.229, 0.224, 0.225)):
    return (im - rgb_mean) / (rgb_std)

def denormalize_np(im, rgb_mean=(0.485, 0.456, 0.406), rgb_std=(0.229, 0.224, 0.225)):
    return im * (rgb_std) + rgb_mean

def denormalize_tensor(im, rgb_mean=(0.485, 0.456, 0.406), rgb_std=(0.229, 0.224, 0.225)):
    mean_tensor = torch.tensor(rgb_mean).reshape(1, -1, 1, 1).to(im.device)
    std_tensor = torch.tensor(rgb_std).reshape(1, -1, 1, 1).to(im.device)
    return im * (std_tensor) + mean_tensor