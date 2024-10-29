import os
import datetime
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import wandb

import options.options as option
from data import create_dataloader, create_dataset
from model import create_model
from loss import FocalLoss
import utils.utils as utils
import data.utils as data_utils

from huggingface_hub import login


abspath = os.path.abspath(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.')
parser.add_argument('-root', type=str, default=None, choices=['.'])
args = parser.parse_args()
opt = option.parse(args.opt, root=args.root)

opt = option.dict_to_nonedict(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % opt['gpu_ids'][0]
device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')

# HF Login to get pretrained weight
# login(opt['token'])

for phase, dataset_opt in opt['datasets'].items():
    if phase=='train': 
        train_set = create_dataset(dataset_opt)
        train_loader = create_dataloader(train_set, dataset_opt, opt, None)
    elif phase=='valid': 
        valid_set = create_dataset(dataset_opt)
        valid_loader = create_dataloader(valid_set, dataset_opt, opt, None)
    elif phase=='test': 
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt, opt, None)
    else:
        raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    
working_dir = os.path.join('.', opt['job_dir'], opt['name'])
os.makedirs(working_dir, exist_ok=True)
    
model = create_model(opt)

# fix and load weight
if opt['path']['pretrain_model'] is not None:
    state_dict = torch.load(opt['path']['pretrain_model'], map_location='cpu')
    current_dict = model.state_dict()
    new_state_dict = state_dict
    # new_state_dict={k:v if v.size()==current_dict[k].size()  else  current_dict[k] for k,v in zip(current_dict.keys(), state_dict.values())}    # fix the size of checkpoint state dict
    _strict=True
    if opt['name'] == 'ProvGigaPath':   # Not load trained weight but the quantized pretrained weight for FM encoder
        new_state_dict = {k: v for k, v in new_state_dict.items() if 'classifier' in k}
        _strict=False
    
    # print(new_state_dict.keys())
    model.load_state_dict(new_state_dict, strict=_strict)  
    print("[INFO] Load weight from:", opt['path']['pretrain_model'])

train_opt = opt['train']

if hasattr(model, "enable_lora_training"):
    model.enable_lora_training()
    
optimizer = utils.create_optimizer(model.parameters(), train_opt)

weight = torch.tensor([0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2]).to(device)
weight = weight / torch.sum(weight)
loss_func = nn.CrossEntropyLoss(weight=weight)
# loss_func = FocalLoss().to(device)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_opt['epochs'], train_opt['eta_min'])

for name, params in model.named_parameters():
    if params.requires_grad:
        print(name)

def train():
    
    model.to(device)
    
    #### Initialization ####
    loss_tracker = utils.MetricTracker('Train Loss')
    acc_tracker = utils.MetricTracker('Train Accuracy')
    best_acc = 0.0
    best_prec = 0.0
    global_step = 0
    #### Start Training ####
    for epoch in range(train_opt['epochs']):
        train_metrics = {}  # reset
        
        # Validation #
        if (epoch % train_opt['val_freq']==0):    
            print("Evaluating...")
            eval_loss, eval_acc, eval_metrics = evaluate()
            
            print(f"[EVAL] Epoch {epoch}|{eval_loss}|{eval_acc}")
            for k, v in eval_metrics.items():
                eval_metrics[k] = np.mean(v / len(valid_loader))
                print(f"Eval {k}: {round(eval_metrics[k], 3)}", end= '|')
                
            if opt['wandb']: 
                wandb.log({f"eval_epoch_{k}": v for k, v in eval_metrics.items()})
            
            if eval_metrics['prec'] > best_prec:
                print(f"[WARN] Save best performance model at epoch {epoch} - step {global_step}!")
                # best_acc = eval_acc.avg
                best_prec = eval_metrics['prec']
                torch.save(model.state_dict(), os.path.join(working_dir, '_best.pt'))
        
            
        # Training Loop #
        model.train()
        all_train_preds = []
        all_train_gts = []
        for im, gt in tqdm(train_loader, total=len(train_loader)):
            batch_size = im.shape[0]
            im = im.to(device)
            gt = gt.to(device)
            
            pred = model(im)
            # loss = loss_func(pred, gt) + 0.5 * regularization(pred, gt)
            loss = loss_func(pred, gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_tracker.update(loss.detach().cpu().item(), batch_size)
            if train_opt['mode']=='segment':
                iou_, prec_, recall_, acc_ = utils.compute_segmentation_metrics(pred, gt)
            elif train_opt['mode']=='classification':
                iou_, prec_, recall_, acc_ = utils.compute_classification_metrics(pred, gt)
                train_metrics['iou'] = train_metrics.get('iou', 0) + np.array(iou_)
                train_metrics['prec'] = train_metrics.get('prec', 0) + np.array(prec_)
                train_metrics['recall'] = train_metrics.get('recall', 0) + np.array(recall_)
                train_metrics['acc'] = train_metrics.get('acc', 0) + np.array(acc_)
                train_metrics['loss'] = train_metrics.get('loss', 0) + loss.detach().cpu().item()

                # all_train_preds.append(pred.clone().detach().cpu())
                # all_train_gts.append(gt.clone().detach().cpu())
            acc_tracker.update(acc_, batch_size)
            
            
            global_step += 1
            
            # Validation #
            if (global_step % train_opt['val_step_freq']==0):    
                print("Evaluating...")
                eval_loss, eval_acc, eval_metrics = evaluate()
                
                print(f"[EVAL] Epoch {epoch}|{eval_loss}|{eval_acc}")
                for k, v in eval_metrics.items():
                    eval_metrics[k] = np.mean(v / len(valid_loader))
                    print(f"Eval {k}: {round(eval_metrics[k].mean(), 3)}", end= '|')
                    
                if opt['wandb']: 
                    wandb.log({f"eval_{k}": v for k, v in eval_metrics.items()})
                
                if eval_metrics['prec'] > best_prec:
                    print(f"[WARN] Save best performance model at epoch {epoch} - step {global_step}!")
                    # best_acc = eval_acc.avg
                    best_prec = eval_metrics['prec']
                    torch.save(model.state_dict(), os.path.join(working_dir, '_best.pt'))
            
        print(f"[Train] Epoch {epoch}|{loss_tracker}|{acc_tracker}")
        
        
        # if train_opt['mode'] == 'classification':
        #     pred = torch.cat(all_train_preds, dim=0)
        #     gt = torch.cat(all_train_gts, dim=0)
        #     # iou_, prec_, recall_, acc_ = utils.compute_classification_metrics(pred, gt)
            
        #     iou_, prec_, recall_, acc_ = utils.compute_classification_metrics(pred, gt)
        #     train_metrics['iou'] = train_metrics.get('iou', 0) + np.array(iou_)*batch_size
        #     train_metrics['prec'] = train_metrics.get('prec', 0) + np.array(prec_)*batch_size
        #     train_metrics['recall'] = train_metrics.get('recall', 0) + np.array(recall_)*batch_size
        #     train_metrics['acc'] = train_metrics.get('acc', 0) + np.array(acc_)*batch_size
        
        for k, v in train_metrics.items():
            train_metrics[k] = np.mean(v / len(train_loader))
            print(f"Train {k}: {round(train_metrics[k], 3)}", end= '|')
            
        if opt['wandb']: wandb.log({f"train_{k}": v for k, v in train_metrics.items()})
        
        lr_scheduler.step()
        
        loss_tracker.reset()
        acc_tracker.reset()
        
        torch.save(model.state_dict(), os.path.join(working_dir, '_last.pt'))
            
    return

def evaluate():
    
    loss_tracker = utils.MetricTracker('Valid Loss')
    acc_tracker = utils.MetricTracker('Valid Accuracy')
    model.to(device)
    model.eval()
    
    metrics = {}
    
    all_preds = []
    all_gts = []
    
    for im, gt in tqdm(valid_loader, total=len(valid_loader)):
        batch_size = im.shape[0]
        im = im.to(device)
        gt = gt.to(device)
        
        with torch.no_grad():
            pred = model(im)
            
        loss = loss_func(pred, gt)
        loss_tracker.update(loss.detach().cpu().item(), batch_size)
        
        if train_opt['mode']=='segment':
            iou_, prec_, recall_, acc_ = utils.compute_segmentation_metrics(pred, gt)
        elif train_opt['mode']=='classification':
            iou_, prec_, recall_, acc_ = utils.compute_classification_metrics(pred, gt)
        else: assert(0), train_opt['mode']
        
        metrics['iou'] = metrics.get('iou', 0) + np.array(iou_)
        metrics['prec'] = metrics.get('prec', 0) + np.array(prec_)
        metrics['recall'] = metrics.get('recall', 0) + np.array(recall_)
        metrics['acc'] = metrics.get('acc', 0) + np.array(acc_)
        metrics['loss'] = metrics.get('loss', 0) + np.array(loss.detach().cpu().item()) 
            # all_preds.append(pred.clone().detach().cpu())
            # all_gts.append(gt.clone().detach().cpu()
        acc_tracker.update(acc_, batch_size)
    
    # print(all_preds)
    # if train_opt['mode']=='classification':
    #     pred = torch.cat(all_preds, dim=0)
    #     gt = torch.cat(all_gts, dim=0)
    #     iou_, prec_, recall_, acc_ = utils.compute_classification_metrics(pred, gt)
    #     metrics['iou'] = metrics.get('iou', 0) + np.array(iou_)
    #     metrics['prec'] = metrics.get('prec', 0) + np.array(prec_)
    #     metrics['recall'] = metrics.get('recall', 0) + np.array(recall_)
    #     metrics['acc'] = metrics.get('acc', 0) + np.array(acc_)
        
    # print(utils.compute_all_metrics(pred, gt))
    
    model.train()
    
    return loss_tracker, acc_tracker, metrics


def visualize(im, gt, pred, im_id):
    
    outdir = './analyze'
    os.makedirs(outdir, exist_ok=True)
    
    np_im = im.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    np_im = data_utils.denormalize_np(np_im)
    print(np.mean(np_im)*255)
    
    gt = gt.detach().cpu().squeeze(0).numpy()   # H, W
    pred = torch.argmax(pred, dim=1)
    pred = pred.detach().cpu().squeeze(0).numpy()   # CxHxW
    # output = np.zeros((gt.shape[0], gt.shape[1], 3))
    # outgt = np.zeros((gt.shape[0], gt.shape[1], 3))
    
    # target_colors = [
    #     [255, 255, 255],
    #     [0, 128, 0],
    #     [255, 143, 204],
    #     [255, 0, 0],
    #     [0, 0, 0],
    #     [165, 42, 42],
    #     [0, 0, 255]]
    
    # for idx, color in enumerate(target_colors):
    #     color = np.array(color)
        
    #     mask = pred == idx
    #     output[mask] = color

    #     mask = gt == idx
    #     outgt[mask] = color
    
    if pred != gt:
        
        # plt.imsave(os.path.join(outdir, f"{im_id}_pred"), output.astype(np.uint8))
        # plt.imsave(os.path.join(outdir, f"{im_id}_gt"), outgt.astype(np.uint8))
        plt.imsave(os.path.join(outdir, f"tpatch_{im_id}.png"), np_im)
        print(gt)
            
    # return output

def test():
    
    loss_tracker = utils.MetricTracker('Test Loss')
    acc_tracker = utils.MetricTracker('Test Accuracy')
    model.to(device)
    model.eval()
    
    metrics = {}
    
    all_preds = []
    all_gts = []
    cnt = 0
    for im, gt in tqdm(test_loader, total=len(test_loader)):
        batch_size = im.shape[0]
        im = im.to(device)
        gt = gt.to(device)
        
        with torch.no_grad():
            pred = model(im)
            
        loss = loss_func(pred, gt)
        loss_tracker.update(loss.detach().cpu().item(), batch_size)
        
        if train_opt['mode']=='segment':
            iou_, prec_, recall_, acc_ = utils.compute_segmentation_metrics(pred, gt)
        elif train_opt['mode']=='classification':
            iou_, prec_, recall_, acc_ = utils.compute_classification_metrics(pred, gt)
        else: assert(0), train_opt['mode']
        acc_tracker.update(acc_, batch_size)
        
        # metrics['iou'] = metrics.get('iou', 0) + np.array(iou_)*batch_size
        # metrics['prec'] = metrics.get('prec', 0) + np.array(prec_)*batch_size
        # metrics['recall'] = metrics.get('recall', 0) + np.array(recall_)*batch_size
        # metrics['acc'] = metrics.get('acc', 0) + np.array(acc_)
        # metrics['loss'] = metrics.get('loss', 0) + np.array(loss.detach().cpu().item())
        all_preds.append(pred.clone().detach().cpu())
        all_gts.append(gt.clone().detach().cpu())
        
        # idx_pred = torch.argmax(pred.cpu().detach(), dim=1).item()
        # if gt.item() != idx_pred:
        # # if gt.item()==3:
        
        #     visualize(im, gt, pred, cnt)
        #     print(idx_pred, gt)
        cnt += 1
    
    pred = torch.cat(all_preds, dim=0)
    gt = torch.cat(all_gts, dim=0)
    iou_, prec_, recall_, acc_ = utils.compute_classification_metrics(pred, gt)
    metrics['iou'] = metrics.get('iou', 0) + np.array(iou_)
    metrics['prec'] = metrics.get('prec', 0) + np.array(prec_)
    metrics['recall'] = metrics.get('recall', 0) + np.array(recall_)
    metrics['acc'] = metrics.get('acc', 0) + np.array(acc_)
        
    for k, v in metrics.items():
        # metrics[k] = np.mean(v)
        print(f"{k}: {metrics[k]}")
        
    print(f"{loss_tracker}|{acc_tracker}")
    
    
    return loss_tracker, acc_tracker
        
    
if __name__ == '__main__':
    if opt['is_train']:
        print("[INFO] Start training...")
        if opt['wandb']:
            wandb.login(key="60fd0a73c2aefc531fa6a3ad0d689e3a4507f51c")
            wandb.init(
                project="BoneTumor",
                name=opt['name'])
        train()
    elif opt['is_test']:
        print("[INFO] Start testing...")
        test()
        
        
        