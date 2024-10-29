import os
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
# import cv2
import torch
import pandas as pd
import torch.nn as nn
# from torchvision.utils import make_grid
# import warnings
# from scipy.special import softmax
import matplotlib.pyplot as plt
# from scipy import stats
from PIL import Image

# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    
def create_optimizer(params, opt):
    return torch.optim.Adam(
        params,
        lr = opt['lr_G'], betas=(opt['beta1'], opt['beta2'])
    )

#######################
# metric
#######################

class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} average: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)
    
# def compute_acc(y_pred, y_true):
#     y_pred = torch.argmax(y_pred, dim=1).cpu().numpy().reshape(-1)
#     y_true = y_true.cpu().numpy().reshape(-1)
#     return accuracy_score(y_true, y_pred)

# def compute_all_metrics(y_pred, y_true):
#     precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
#     print('precision: {}'.format(precision))
#     print('recall: {}'.format(recall))
#     print('fscore: {}'.format(fscore))
#     print('support: {}'.format(support))
    
############################################

def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        print(log_file)
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


####################
# image convert
####################

# def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
#     '''
#     Converts a torch Tensor into an image Numpy array
#     Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
#     Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
#     '''
#     tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
#     tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
#     n_dim = tensor.dim()
#     if n_dim == 4:
#         n_img = len(tensor)
#         img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
#         img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
#     elif n_dim == 3:
#         img_np = tensor.numpy()
#         img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
#     elif n_dim == 2:
#         img_np = tensor.numpy()
#     else:
#         raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
#     if out_type == np.uint8:
#         img_np = (img_np * 255.0).round()
#         # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
#     return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    #**
    return
    # cv2.imwrite(img_path, img)

class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()
        
######### Image split and combine ##########    

def crop(img, crop_sz, step):
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
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
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            lr_list.append(crop_img)
    h=x + crop_sz
    w=y + crop_sz
    return lr_list, num_h, num_w, h, w

def combine(sr_list, num_h, num_w, h, w, patch_size, step, channel=3):
    index=0
    sr_img = np.zeros((h, w, channel), 'float32')
    # print(h, w, num_h, num_w, channel)
    for i in range(num_h):
        for j in range(num_w):
            sr_subim = sr_list[index]
                
            # bg = np.ones((patch_size, patch_size, channel), 'float32')
            # r, c, _ = sr_subim.shape
            # bg[:r, :c, :] = sr_subim
            # sr_subim = bg.astype(np.float32)
            
            sr_img[i*step: i*step+patch_size, j*step: j*step+patch_size,:]+=sr_subim
            index+=1
            
    # sr_img=sr_img.astype('float32')

    for j in range(1,num_w):
        sr_img[:,j*step:j*step+(patch_size-step),:]/=2

    for i in range(1,num_h):
        sr_img[i*step:i*step+(patch_size-step),:,:]/=2
    return sr_img

def clustering_pytorch(features, num_labels, niters=100):
    # Initialize centroids randomly
    # features = torch.cat(feature_list, dim=0)
    centroids = features[torch.randperm(features.size(0))[:num_labels]]
    
    for _ in range(niters):
        # Calculate distances from data points to centroids
        distances = torch.cdist(features, centroids)
    
        # Assign each data point to the closest centroid
        _, labels = torch.min(distances, dim=1)
    
        # Update centroids by taking the mean of data points assigned to each centroid
        for i in range(num_labels):
            if torch.sum(labels == i) > 0:
                centroids[i] = torch.mean(features[labels == i], dim=0)
                
    return centroids, labels


def compute_segmentation_metrics(preds, targets):
    """
    Calculate IoU, Precision, and Recall for multi-class segmentation.
    
    Args:
        preds: predicted segmentation maps (B, C, H, W) -> usually softmax output or logits
        targets: ground truth segmentation maps (B, H, W) -> contains class indices (not one-hot)
        
    Returns:
        iou_per_class: IoU for each class
        precision_per_class: Precision for each class
        recall_per_class: Recall for each class
    """
    num_classes = preds.shape[1]
    # Step 1: Convert logits or softmax to predicted class labels (B, H, W)
    preds = torch.argmax(preds, dim=1)  # (B, H, W)
    
    # Initialize metrics for each class
    iou_per_class = []
    precision_per_class = []
    recall_per_class = []
    
    # Step 2: Calculate metrics for each class
    for class_idx in range(num_classes):
        # True Positives, False Positives, and False Negatives for each class
        TP = ((preds == class_idx) & (targets == class_idx)).sum(dim=(1, 2))  # (B,)
        FP = ((preds == class_idx) & (targets != class_idx)).sum(dim=(1, 2))  # (B,)
        FN = ((preds != class_idx) & (targets == class_idx)).sum(dim=(1, 2))  # (B,)
        
        if (TP + FP + FN).sum() == 0:
            iou_per_class.append(1.0)  # Perfect IoU
            precision_per_class.append(1.0)  # Perfect Precision
            recall_per_class.append(1.0)  # Perfect Recall
        else:
            # Avoid division by zero by adding a small epsilon
            epsilon = 1e-7
            
            # IoU = TP / (TP + FP + FN)
            iou = TP / (TP + FP + FN + epsilon)
            iou_per_class.append(iou.mean().item())
            
            # Precision = TP / (TP + FP)
            precision = TP / (TP + FP + epsilon)
            precision_per_class.append(precision.mean().item())
            
            # Recall = TP / (TP + FN)
            recall = TP / (TP + FN + epsilon)
            recall_per_class.append(recall.mean().item())
        
    correct_pixels = (preds == targets).sum(dim=(1, 2))  # Total correctly predicted pixels per image
    total_pixels = torch.tensor(targets.shape[-2] * targets.shape[-1], dtype=torch.float32)  # Total pixels in one image
    accuracy = correct_pixels.float() / total_pixels.float()  # Pixel accuracy for each image in the batch
    accuracy = accuracy.mean().item()  # Average accuracy across the batch
    
    return iou_per_class, precision_per_class, recall_per_class, accuracy

def compute_classification_metrics(preds, targets, drop_last=False, inv_mask=None):
    """
    Calculate IoU, Precision, and Recall for multi-class segmentation.
    
    Args:
        preds: predicted segmentation maps (B, C, H, W) -> usually softmax output or logits
        targets: ground truth segmentation maps (B, H, W) -> contains class indices (not one-hot)
        
    Returns:
        iou_per_class: IoU for each class
        precision_per_class: Precision for each class
        recall_per_class: Recall for each class
    """
    # Step 1: Convert logits or softmax to predicted class labels (B, H, W)
    num_classes = preds.shape[-1]
    preds = torch.argmax(preds, dim=-1)
    
    # Initialize metrics for each class
    iou_per_class = []
    precision_per_class = []
    recall_per_class = []
    
    # Step 2: Calculate metrics for each class
    if drop_last: 
        num_classes = num_classes - 1
    if inv_mask is not None:
        # print(preds.shape, inv_mask.shape)
        preds = preds.masked_select(inv_mask)
        targets = targets.masked_select(inv_mask)
    for class_idx in range(num_classes):
        # True Positives, False Positives, and False Negatives for each class
        
        TP = ((preds == class_idx) & (targets == class_idx)).sum()  # (B,)
        FP = ((preds == class_idx) & (targets != class_idx)).sum()  # (B,)
        FN = ((preds != class_idx) & (targets == class_idx)).sum()  # (B,)
        
        if (TP + FP + FN).sum() == 0:
            iou_per_class.append(1.0)  # Perfect IoU
            precision_per_class.append(1.0)  # Perfect Precision
            recall_per_class.append(1.0)  # Perfect Recall
        else:
            # Avoid division by zero by adding a small epsilon
            epsilon = 1e-7
            
            # IoU = TP / (TP + FP + FN)
            iou = TP / (TP + FP + FN + epsilon)
            iou_per_class.append(iou.mean().item())
            
            # Precision = TP / (TP + FP)
            precision = TP / (TP + FP + epsilon)
            precision_per_class.append(precision.mean().item())
            
            # Recall = TP / (TP + FN)
            recall = TP / (TP + FN + epsilon)
            recall_per_class.append(recall.mean().item())
        
    correct_pixels = (preds == targets).float() # Total correctly predicted pixels per image
    # total_pixels = torch.tensor(targets.shape[-2] * targets.shape[-1], dtype=torch.float32)  # Total pixels in one image
    # accuracy = correct_pixels.float() / total_pixels.float()  # Pixel accuracy for each image in the batch
    accuracy = correct_pixels.mean().item()  # Average accuracy across the batch
    
    return iou_per_class, precision_per_class, recall_per_class, accuracy
    