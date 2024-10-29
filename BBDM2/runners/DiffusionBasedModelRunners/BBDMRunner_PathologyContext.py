import os

import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import datetime
import time
import os
import traceback
import wandb
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from model.BrownianBridge.LatentBrownianBridgeModel_PathologyContext import LatentBrownianBridgeModel_Pathology
from model.BrownianBridge.BrownianBridgeModel_PathologyContext import BrownianBridgeModel_Pathology

from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from tqdm.autonotebook import tqdm
from runners.base.EMA import EMA
from runners.utils import make_save_dirs, make_dir, get_dataset, remove_file


@Registers.runners.register_with_name('BBDMRunner_PathologyContext')
class BBDMRunner_PathologyContext(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.use_wandb = config.training.wandb
        if self.use_wandb: self.init_wandb(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM-Pathology":
            bbdmnet = LatentBrownianBridgeModel_Pathology(config.model).to(config.training.device[0])
        elif config.model.model_type=='BBDM-Pathology':
            bbdmnet = BrownianBridgeModel_Pathology(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet
    
    def init_wandb(self, config):
        print('[INFO] Initiate wandb logging')
        wandb.login(key=config.training.wandb_key)
        wandb.init(
            project="BoneTumor",
            group = "Phase2_BBDM",
            name = config.data.dataset_name
        )

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
                # self.net.cont_latent_std = states['cont_latent_std'].to(self.config.training.device[0])
                # self.net.cont_latent_mean = states['cont_latent_mean'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        self.logger("Total Number of parameter: %.2fM" % (total_num / 1e6))
        self.logger("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler)
)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
                # model_states['cont_latent_mean'] = self.net.module.cont_latent_mean
                # model_states['cont_latent_std'] = self.net.module.cont_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
                # model_states['cont_latent_mean'] = self.net.cont_latent_mean
                # model_states['cont_latent_std'] = self.net.cont_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        total_cont_mean = None
        total_cont_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None, total_cont_mean=None):
            (x, x_name), (x_cond, x_cond_name), (x_cont, x_cont_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])
            x_cont = x_cont.to(self.config.training.device[0])

            x_latent = self.net.encode(x, type='ori', normalize=False)
            x_cond_latent = self.net.encode(x_cond, type='cond', normalize=False)
            # x_cont_latent = self.net.encoder(x_cont, type='cont', normalize=False)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            
            # x_cont_mean = x_cont_latent.mean(axis=[0,2,3], keepdim=True)
            # total_cont_mean = x_cont_mean if total_cont_mean is None else x_cont_mean + total_cont_mean
            
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, cont_latent_mean=None,
                    total_ori_var=None, total_cond_var=None, total_cont_var=None):
            (x, x_name), (x_cond, x_cond_name), (x_cont, x_cont_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])
            x_cont = x_cont.to(self.config.training.device[0])

            x_latent = self.net.encode(x, type='ori', normalize=False)
            x_cond_latent = self.net.encode(x_cond, type='cond', normalize=False)
            # x_cont_latent = self.net.encode(x_cont, type='cont', normalize=False)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            
            # x_cont_var = ((x_cont_latent - cont_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            # total_cont_var = x_cont_var if total_cont_var is None else x_cont_var + total_cont_var
            
            return total_ori_var, total_cond_var

        self.logger(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean, total_cont_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        # cont_latent_mean = total_cont_mean / batch_count
        # self.net.cont_latent_mean = cont_latent_mean.cpu()

        self.logger(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var,)
            # break

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count
        # cont_latent_var = total_cont_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        # self.net.cont_latent_std = torch.sqrt(cont_latent_var)
        
        self.logger(self.net.ori_latent_mean)
        self.logger(self.net.ori_latent_std)
        self.logger(self.net.cond_latent_mean)
        self.logger(self.net.cond_latent_std)
        # self.logger(self.net.cont_latent_mean)
        # self.logger(self.net.cont_latent_std)

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        (x, x_name), (x_cond, x_cond_name), (x_cont, x_cont_name) = batch
        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])
        x_cont = x_cont.to(self.config.training.device[0])

        loss, additional_info = net(x, x_cond, context=x_cont)
        
        if write and self.is_main_process:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if additional_info.__contains__('recloss_noise'):
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
            if additional_info.__contains__('recloss_xy'):
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
                
        x_latent_recon = additional_info['x0_recon'].cpu()
        x_latent = additional_info['x0'].cpu()
        diff = (x_latent_recon - x_latent).pow(2).mean()
        psnr = (10 * torch.log10(255**2 / diff)).item()
        
        if write and self.is_main_process:
            self.writer.add_scalar(f'psnr/{stage}', psnr, step)
        
        del additional_info, x, x_cond, x_cont
                
        return loss, psnr

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        reverse_sample_path = make_dir(os.path.join(sample_path, 'reverse_sample'))
        reverse_one_step_path = make_dir(os.path.join(sample_path, 'reverse_one_step_samples'))

        print(sample_path)

        (x, x_name), (x_cond, x_cond_name), (x_cont, x_cont_name) = batch

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[0:batch_size].to(self.config.training.device[0])
        x_cont = x_cont[0:batch_size].to(self.config.training.device[0])

        grid_size = 4

        # samples, one_step_samples = net.sample(x_cond,
        #                                        clip_denoised=self.config.testing.clip_denoised,
        #                                        sample_mid_step=True)
        # self.save_images(samples, reverse_sample_path, grid_size, save_interval=200,
        #                  writer_tag=f'{stage}_sample' if stage != 'test' else None)
        #
        # self.save_images(one_step_samples, reverse_one_step_path, grid_size, save_interval=200,
        #                  writer_tag=f'{stage}_one_step_sample' if stage != 'test' else None)
        #
        # sample = samples[-1]
        sample = net.sample(x_cond, x_cont, clip_denoised=self.config.testing.clip_denoised).to('cpu')
        
        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x_cond.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'condition.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')
            
        image_grid = get_image_grid(x_cont[:, 3:, ...].to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'context.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_context', image_grid, self.global_step, dataformats='HWC')

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        condition_path = make_dir(os.path.join(sample_path, f'condition'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        cont_path = make_dir(os.path.join(sample_path, 'context'))
        result_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        batch_size = self.config.data.test.batch_size
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num
        for test_batch in pbar:
            (x, x_name), (x_cond, x_cond_name), (x_cont, x_cont_name) = test_batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])
            x_cont = x_cont.to(self.config.training.device[0])

            for j in range(sample_num):
                sample = net.sample(x_cond, x_cont, clip_denoised=False)
                # sample = net.sample_vqgan(x)
                for i in range(batch_size):
                    condition = x_cond[i].detach().clone()
                    gt = x[i]
                    cont = x_cont[i].detach().clone()
                    result = sample[i]
                    if j == 0:
                        save_single_image(condition, condition_path, f'{x_cond_name[i]}.png', to_normal=to_normal)
                        save_single_image(gt, gt_path, f'{x_name[i]}.png', to_normal=to_normal)
                        save_single_image(cont, cont_path, f'{x_cont_name[i]}.png', to_normal=to_normal)
                    if sample_num > 1:
                        result_path_i = make_dir(os.path.join(result_path, x_name[i]))
                        save_single_image(result, result_path_i, f'output_{j}.png', to_normal=to_normal)
                    else:
                        save_single_image(result, result_path, f'{x_name[i]}.png', to_normal=to_normal)
                        
                        
    @torch.no_grad()
    def validation_epoch(self, val_loader, epoch):
        self.apply_ema()
        self.net.eval()

        pbar = tqdm(val_loader, total=len(val_loader), smoothing=0.01, disable=not self.is_main_process)
        step = 0
        loss_sum = 0.
        dloss_sum = 0.
        psnr_sum = 0.
        for val_batch in pbar:
            loss, psnr = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=0,
                                stage='val',
                                write=False)
            loss_sum += loss.cpu().detach()
            psnr_sum += psnr
            if len(self.optimizer) > 1:
                loss = self.loss_fn(net=self.net,
                                    batch=val_batch,
                                    epoch=epoch,
                                    step=step,
                                    opt_idx=1,
                                    stage='val',
                                    write=False)
                dloss_sum += loss
            step += 1
        average_loss = loss_sum / step
        average_psnr = psnr_sum / step
        if self.is_main_process:
            self.writer.add_scalar(f'val_epoch/loss', average_loss, epoch)
            self.writer.add_scalar(f'val_epoch/psnr', average_psnr, epoch)
            if len(self.optimizer) > 1:
                average_dloss = dloss_sum / step
                self.writer.add_scalar(f'val_dloss_epoch/loss', average_dloss, epoch)
        self.restore_ema()
        
        del loss
        
        return average_loss, average_psnr
                        
    def train(self):
        self.logger(self.__class__.__name__)

        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_sampler = None
        val_sampler = None
        test_sampler = None
        if self.config.training.use_DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.config.data.train.batch_size,
                                      num_workers=8,
                                      drop_last=True,
                                      sampler=train_sampler)
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.config.data.val.batch_size,
                                    num_workers=8,
                                    drop_last=True,
                                    sampler=val_sampler)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     num_workers=8,
                                     drop_last=True,
                                     sampler=test_sampler)
        else:
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.config.data.train.batch_size,
                                      shuffle=self.config.data.train.shuffle,
                                      num_workers=8,
                                      drop_last=True)
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.config.data.val.batch_size,
                                    shuffle=self.config.data.val.shuffle,
                                    num_workers=8,
                                    drop_last=True)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=8,
                                     drop_last=True)

        epoch_length = len(train_loader)
        start_epoch = self.global_epoch
        self.logger(f"start training {self.config.model.model_name} on {self.config.data.dataset_name}, {len(train_loader)} iters per epoch")

        try:
            accumulate_grad_batches = self.config.training.accumulate_grad_batches
            for epoch in range(start_epoch, self.config.training.n_epochs):
                # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
                if self.global_step > self.config.training.n_steps:
                    break

                if self.config.training.use_DDP:
                    train_sampler.set_epoch(epoch)
                    val_sampler.set_epoch(epoch)
                                
                #----------------- validation
                if epoch % self.config.training.validation_interval == 0 or (
                        epoch + 1) == self.config.training.n_epochs:
                    # if self.is_main_process == 0:
                    with torch.no_grad():
                        self.logger("validating epoch...")
                        average_loss, average_psnr = self.validation_epoch(val_loader, epoch)
                        # torch.cuda.empty_cache()
                        self.logger("validating epoch success")
                        
                        val_metric = {
                            'val_loss': average_loss,
                            'val_psnr': average_psnr,
                        }
                        if self.use_wandb: wandb.log(val_metric)
                        
                # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
                        
                #-------------------- save checkpoint
                if epoch % self.config.training.save_interval == 0 or \
                        (epoch + 1) == self.config.training.n_epochs or \
                        self.global_step > self.config.training.n_steps:
                    if self.is_main_process:
                        with torch.no_grad():
                            self.logger("saving latest checkpoint...")
                            self.on_save_checkpoint(self.net, train_loader, val_loader, epoch, self.global_step)
                            model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')

                            # save latest checkpoint
                            temp = 0
                            while temp < epoch + 1:
                                remove_file(os.path.join(self.config.result.ckpt_path, f'latest_model_{temp}.pth'))
                                remove_file(
                                    os.path.join(self.config.result.ckpt_path, f'latest_optim_sche_{temp}.pth'))
                                temp += 1
                            torch.save(model_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'latest_model_{epoch + 1}.pth'))
                            torch.save(optimizer_scheduler_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'latest_optim_sche_{epoch + 1}.pth'))
                            torch.save(model_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'last_model.pth'))
                            torch.save(optimizer_scheduler_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'last_optim_sche.pth'))

                            # save top_k checkpoints
                            model_ckpt_name = f'top_model_epoch_{epoch + 1}.pth'
                            optim_sche_ckpt_name = f'top_optim_sche_epoch_{epoch + 1}.pth'

                            if self.config.args.save_top:
                                print("save top model start...")
                                top_key = 'top'
                                if top_key not in self.topk_checkpoints:
                                    print('top key not in topk_checkpoints')
                                    self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                      'psnr': average_psnr,
                                                                      'model_ckpt_name': model_ckpt_name,
                                                                      'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                    print(f"saving top checkpoint: average_loss={average_loss} average_psnr={average_psnr} epoch={epoch + 1}")
                                    torch.save(model_states,
                                               os.path.join(self.config.result.ckpt_path, model_ckpt_name))
                                    torch.save(optimizer_scheduler_states,
                                               os.path.join(self.config.result.ckpt_path, optim_sche_ckpt_name))
                                else:
                                    # if average_loss < self.topk_checkpoints[top_key]["loss"]:
                                    if average_psnr > self.topk_checkpoints[top_key]['psnr']:
                                        print("remove " + self.topk_checkpoints[top_key]["model_ckpt_name"])
                                        remove_file(os.path.join(self.config.result.ckpt_path,
                                                                 self.topk_checkpoints[top_key]['model_ckpt_name']))
                                        remove_file(os.path.join(self.config.result.ckpt_path,
                                                                 self.topk_checkpoints[top_key]['optim_sche_ckpt_name']))

                                        print(
                                            f"saving top checkpoint: average_loss={average_loss} average_psnr={average_psnr} epoch={epoch + 1}")

                                        self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                          "psnr": average_psnr,
                                                                          'model_ckpt_name': model_ckpt_name,
                                                                          'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                        torch.save(model_states,
                                                   os.path.join(self.config.result.ckpt_path, model_ckpt_name))
                                        torch.save(optimizer_scheduler_states,
                                                   os.path.join(self.config.result.ckpt_path, optim_sche_ckpt_name))
                                        
                # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
                                        
                if self.config.training.use_DDP:
                    dist.barrier()
                # torch.cuda.empty_cache()
                
                #--------------- start training
                pbar = tqdm(train_loader, total=len(train_loader), smoothing=0.01, disable=not self.is_main_process)
                self.global_epoch = epoch
                start_time = time.time()
                
                train_losses = []
                train_psnrs = []
                for train_batch in pbar:
                    self.global_step += 1
                    self.net.train()

                    losses = []
                    for i in range(len(self.optimizer)):
                        # pdb.set_trace()
                        loss, psnr = self.loss_fn(net=self.net,
                                            batch=train_batch,
                                            epoch=epoch,
                                            step=self.global_step,
                                            opt_idx=i,
                                            stage='train')
                        
                        # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())

                        loss.backward()
                        if self.global_step % accumulate_grad_batches == 0:
                            self.optimizer[i].step()
                            self.optimizer[i].zero_grad()
                            if self.scheduler is not None:
                                self.scheduler[i].step(loss)
                        if self.config.training.use_DDP:
                            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
                        losses.append(loss.detach().cpu().mean())
                        train_losses.append(loss.detach().cpu().mean())
                        train_psnrs.append(psnr)

                    if self.use_ema and self.global_step % (self.update_ema_interval*accumulate_grad_batches) == 0:
                        self.step_ema()

                    if len(self.optimizer) > 1:
                        pbar.set_description(
                            (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss-1: {losses[0]:.4f} loss-2: {losses[1]:.4f}'
                            )
                        )
                    else:
                        pbar.set_description(
                            (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss: {losses[0]:.4f}'
                            )
                        )
                    # torch.cuda.empty_cache()
                    with torch.no_grad():
                        if self.global_step % 50 == 0:
                            val_batch = next(iter(val_loader))
                            self.validation_step(val_batch=val_batch, epoch=epoch, step=self.global_step)

                        if self.global_step % int(self.config.training.sample_interval * epoch_length) == 0:
                        # if self.global_step % int(self.config.training.sample_interval) == 0:
                            # val_batch = next(iter(val_loader))
                            # self.validation_step(val_batch=val_batch, epoch=epoch, step=self.global_step)

                            if self.is_main_process:
                                val_batch = next(iter(val_loader))
                                self.sample_step(val_batch=val_batch, train_batch=train_batch)
                                # torch.cuda.empty_cache()

                end_time = time.time()
                elapsed_rounded = int(round((end_time-start_time)))
                self.logger("training time: " + str(datetime.timedelta(seconds=elapsed_rounded)))
                
                train_loss = np.mean(train_losses)
                train_psnr = np.mean(train_psnrs)
                train_metrics = {'train_loss': train_loss,
                                 'train_psnr': train_psnr}
                if self.use_wandb:
                    wandb.log(train_metrics)
                
        except BaseException as e:
            if self.is_main_process == 0:
                print("exception save model start....")
                print(self.__class__.__name__)
                model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='exception')
                torch.save(model_states,
                           os.path.join(self.config.result.ckpt_path, f'last_model.pth'))
                torch.save(optimizer_scheduler_states,
                           os.path.join(self.config.result.ckpt_path, f'last_optim_sche.pth'))

                print("exception save model success!")

            print('str(Exception):\t', str(Exception))
            print('str(e):\t\t', str(e))
            print('repr(e):\t', repr(e))
            print('traceback.print_exc():')
            traceback.print_exc()
            print('traceback.format_exc():\n%s' % traceback.format_exc())

    @torch.no_grad()
    def test(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        if test_dataset is None:
            test_dataset = val_dataset
        # test_dataset = val_dataset
        if self.config.training.use_DDP:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=True,
                                     sampler=test_sampler)
        else:
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=True)

        if self.use_ema:
            self.apply_ema()

        self.net.eval()
        if self.config.args.sample_to_eval:
            sample_path = self.config.result.sample_to_eval_path
            if self.config.training.use_DDP:
                self.sample_to_eval(self.net.module, test_loader, sample_path)
            else:
                self.sample_to_eval(self.net, test_loader, sample_path)
        else:
            test_iter = iter(test_loader)
            for i in tqdm(range(1), initial=0, dynamic_ncols=True, smoothing=0.01):
                test_batch = next(test_iter)
                sample_path = os.path.join(self.config.result.sample_path, str(i))
                if self.config.training.use_DDP:
                    self.sample(self.net.module, test_batch, sample_path, stage='test')
                else:
                    self.sample(self.net, test_batch, sample_path, stage='test')

