import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from model.bbdm.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.bbdm.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.bbdm.VQGAN.vqgan import VQModel


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBrownianBridgeModel_Pathology(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        
        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**model_config['CondStageParams'])
        else:
            raise NotImplementedError
        
        assert(hasattr(self, 'cond_stage_model')), "Cond Stage Model initialization failed"
        
        self.vqgan = VQModel(**model_config['VQGAN']['params']).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config['VQGAN']['params']['ckpt_path']}")


        

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x, x_cond, context=None):
        with torch.no_grad():
            x_latent = self.encode(x, type='ori')
            x_cond_latent = self.encode(x_cond, type='cond')
            # x_cont_latent = self.encoder(x_cont, type='cont')
        context = self.get_cond_stage_context(context)
        return super().forward(x_latent.detach(), x_cond_latent.detach(), context)

    def get_cond_stage_context(self, x_cont):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cont)
            # context = self.encode(x_cont, type='cont')
            if self.condition_key == 'first_stage':
                context = context[1].detach()
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x, type="cond", normalize=None):
        normalize = self.model_config['normalize_latent'] if normalize is None else normalize
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config['latent_before_quant_conv']:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if type=='cond':
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            elif type=='ori':
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
            elif type=='cont':
                x_latent = (x_latent - self.cont_latent_mean) / self.cont_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, type='cond', normalize=None):
        normalize = self.model_config['normalize_latent'] if normalize is None else normalize
        if normalize:
            if type=='cond':
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            elif type=='ori':
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
            elif type=='cont':
                x_latent = x_latent * self.cont_latent_std + self.cont_latent_std
        model = self.vqgan
        if self.model_config['latent_before_quant_conv']:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out

    @torch.no_grad()
    def sample(self, x_cond, x_cont, clip_denoised=False, sample_mid_step=False):
        x_cond_latent = self.encode(x_cond, type='cond')
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     context=self.get_cond_stage_context(x_cont),
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop(y=x_cond_latent,
                                      context=self.get_cond_stage_context(x_cont),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            out = self.decode(x_latent, type='ori')
            return out
        
    @torch.no_grad()
    def sample_infer(self, x_cond, x_cont, clip_denoised=False, sample_mid_step=False):
        x_cond_latent = self.encode(x_cond, type='cond')
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     context=self.get_cond_stage_context(x_cont),
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            out_samples = []
            for i in range(len(temp)):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in range(len(one_step_temp)):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop_eval(y=x_cond_latent,
                                      context=self.get_cond_stage_context(x_cont),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            out = self.decode(x_latent, type='ori')
            return out
        
    @torch.no_grad()
    def sample_eval(self, x_cond, x_cont, clip_denoised=False, sample_mid_step=False):
        x_cond_latent = self.encode(x_cond, type='cond')
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     context=self.get_cond_stage_context(x_cont),
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop_eval(y=x_cond_latent,
                                      context=self.get_cond_stage_context(x_cont),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            # out = self.decode(x_latent, type='ori')
            return x_latent
        
    @torch.no_grad()
    def p_sample_loop_eval(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            # for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
            for i in range(len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in range(len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec

    # @torch.no_grad()
    # def reverse_sample(self, x, skip=False):
    #     x_ori_latent = self.vqgan.encoder(x)
    #     temp, _ = self.brownianbridge.reverse_p_sample_loop(x_ori_latent, x, skip=skip, clip_denoised=False)
    #     x_latent = temp[-1]
    #     x_latent = self.vqgan.quant_conv(x_latent)
    #     x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
    #     out = self.vqgan.decode(x_latent_quant)
    #     return out
