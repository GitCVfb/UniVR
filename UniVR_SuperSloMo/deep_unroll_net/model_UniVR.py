import sys
sys.path.append('gmflow')
from gmflow import GMFlow

import random
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from package_core.net_basics import *
from package_core.model_base import *
from package_core.losses import *
from package_core.flow_utils import *
from package_core.image_proc import *
from flow_loss import *

from network_UniVR import UniVR

class ModelUniVR(ModelBase):
    def __init__(self, opts):
        super(ModelUniVR, self).__init__()
        
        self.opts = opts
        
        # create networks
        self.model_names=['flow', 'vfi']
        
        self.net_flow = GMFlow(feature_channels=128,
                               num_scales=1,
                               upsample_factor=8,
                               num_head=1,
                               attention_type='swin',
                               ffn_dim_expansion=4,
                               num_transformer_layers=6,
                               ).cuda()
        
        self.net_vfi = UniVR().cuda()
        #self.net_flow = torch.nn.DataParallel(UniVR()).cuda()
        
        #self.print_networks(self.net_flow)
        #self.print_networks(self.net_vfi)
        
        # load in initialized network parameters
        if opts.test_pretrained_VFI:
            self.load_pretrained_OF_model(opts.model_label, self.opts.log_dir)#load pretrained optical flow model directly
            self.load_pretrained_GS_model(opts.model_label, self.opts.log_dir)#load pretrained GS-based VFI model directly
        elif not opts.is_training or opts.continue_train:
            self.load_checkpoint(opts.model_label)
        else:
            self.load_pretrained_OF_model(self.opts.model_label_pretrained_GS, self.opts.log_dir_pretrained_GS)#load pretrained optical flow model directly
            self.load_pretrained_GS_model(self.opts.model_label_pretrained_GS, self.opts.log_dir_pretrained_GS)#load pretrained GS-based VFI model
        
        if self.opts.is_training:
            # initialize optimizers
            
            self.optimizer_G = torch.optim.Adam([{'params': self.net_vfi.parameters()}], lr=opts.lr)
            #self.optimizer_G = torch.optim.Adam([{'params': self.net_flow.parameters(), 'lr': 1e-5}, {'params': self.net_vfi.parameters()}], lr=opts.lr)
            
            self.optimizer_names = ['G']
            self.build_lr_scheduler()
            
            # create losses
            self.loss_fn_perceptual = PerceptualLoss(loss=nn.L1Loss())
            self.loss_fn_L1 = L1Loss()
            self.loss_fn_tv2 = VariationLoss(nc=2)
            
            ###Initializing VGG16 model for perceptual loss
            self.MSE_LossFn = nn.MSELoss()
            vgg16 = torchvision.models.vgg16(pretrained=True)
            self.vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
            self.vgg16_conv_4_3.to('cuda')
            for param in self.vgg16_conv_4_3.parameters():
                param.requires_grad = False

    def set_input(self, _input):
        im_rs, im_gs, gt_flow, im_gs_f = _input
        self.im_rs = im_rs.cuda()
        self.im_gs = im_gs
        self.gt_flow = gt_flow
        self.im_gs_f = im_gs_f

        if self.im_gs is not None:
            self.im_gs = self.im_gs.cuda()
        if self.im_gs_f is not None:
            self.im_gs_f = self.im_gs_f.cuda()
        if self.gt_flow is not None:
            self.gt_flow = self.gt_flow.cuda()

    def forward(self, time, gamma):# -gamma/2 <= time <= 1 - gamma/2, 0 <= gamma <= s
        im_rs0 = self.im_rs[:,0:3,:,:].clone()
        im_rs1 = self.im_rs[:,3:6,:,:].clone()

        results_dict = self.net_flow(im_rs0, im_rs1,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1],
                                 pred_bidir_flow=True,
                                 )

        flow_bidir = results_dict['flow_preds'][-1]  # [2*B, 2, H, W]
        #print(flow_bidir.size())
        
        gs_t_final, imgs_warp, flow = self.net_vfi(self.im_rs, flow_bidir, time, gamma)
        
        if self.opts.is_training:
            return gs_t_final, imgs_warp, flow
        
        return gs_t_final


    def optimize_parameters(self):
        
        #===========================================================#
        #                   Initialize losses                       #
        #===========================================================#
        self.loss_L1 = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_perceptual = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_warp = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_flow_smoothness = torch.tensor([0.], requires_grad=True).cuda().float()

        timesteps = [-self.opts.gamma/2, 0, 1-self.opts.gamma/2]#train RSSR for Carla-RS and Fastec-RS
        if self.opts.dataset_type=='BSRSC':
            timesteps = [1-self.opts.gamma/2]#train BS-RSC for RS correction correspinding to the middle scanline of the second RS frame
        
        for tt in range(len(timesteps)):
            if timesteps[tt] == 0:
                im_gt = self.im_gs_f[:,3:6,:,:].clone()
            if timesteps[tt] == -self.opts.gamma/2:
                im_gt = self.im_gs[:,0:3,:,:].clone()
            if timesteps[tt] == 1-self.opts.gamma/2:
                im_gt = self.im_gs[:,3:6,:,:].clone()
            
            pred_im, imgs_warp, flow = self.forward(timesteps[tt], self.opts.gamma)
            #===========================================================#
            #                       Compute losses                      #
            #===========================================================#
            gs_gt = im_gt.clone()
            rs_gt = self.im_rs.clone()
            
            num_imgs_warp = 2 
            for i in range(num_imgs_warp):
                self.loss_warp += self.charbonier_loss(imgs_warp[i], im_gt, mean=True) * self.opts.lamda_L1 *5.0
            
            num_flow = 2 
            for i in range(num_flow):
                self.loss_flow_smoothness += self.opts.lamda_flow_smoothness * self.loss_fn_tv2(flow[i], mean=True)
            
            self.loss_L1 += self.charbonier_loss(pred_im, im_gt, mean=True) * self.opts.lamda_L1 *10.0#Charbonnier
            self.loss_perceptual += self.loss_fn_perceptual.get_loss(pred_im, im_gt) * self.opts.lamda_perceptual
            
        self.loss_L1 = self.loss_L1 / len(timesteps)
        self.loss_perceptual = self.loss_perceptual / len(timesteps)
        self.loss_warp = self.loss_warp / (len(timesteps)*num_imgs_warp)
        self.loss_flow_smoothness = self.loss_flow_smoothness / (len(timesteps)*num_flow)

        # sum them up
        self.loss_G = self.loss_L1 +\
                        self.loss_perceptual +\
                        self.loss_flow_smoothness +\
                        self.loss_warp
                        
        # Optimize 
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

    def warp(self, img, flow):
        B, _, H, W = flow.shape
        xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xx, yy], 1).to(img)
        flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
        grid_ = (grid + flow_).permute(0, 2, 3, 1)
        output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
        return output

    def charbonier_loss(self, pred_im, im_gt, epsilon=0.001, mean=True):
        x = pred_im - im_gt
        loss = torch.mean(torch.sqrt(x ** 2 + epsilon ** 2))
        return loss
    
    # save networks to file 
    def save_checkpoint(self, label):
        self.save_network(self.net_flow, 'flow', label, self.opts.log_dir)
        self.save_network(self.net_vfi,  'vfi',  label, self.opts.log_dir)

    def load_checkpoint(self, label):
        self.load_network(self.net_flow, 'flow', label, self.opts.log_dir)
        self.load_network(self.net_vfi,  'vfi',  label, self.opts.log_dir)

    def load_pretrained_GS_model(self, label, save_dir):
        def convert(param):
            return {
            "UVR."+k: v
                for k, v in param.items()
            }
        
        def convert2(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        
        save_filename = '%s_net_%s.ckpt' % (label, 'vfi')
        save_path = os.path.join(save_dir, save_filename)
        print('load model from ', save_path)

        checkpoint = torch.load(save_path)
        weights = checkpoint['state_dictAT'] if 'state_dictAT' in checkpoint else checkpoint
        
        #self.net_vfi.load_state_dict(convert(torch.load('{}/pre_net_vfi.pth'.format(save_dir))))
        #self.net_vfi.load_state_dict(torch.load(save_path), False)
        self.net_vfi.load_state_dict((convert(weights)))
        
    def load_pretrained_OF_model(self, label, save_dir):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }       
        save_filename = '%s_net_%s.pth' % (label, 'flow')
        save_path = os.path.join(save_dir, save_filename)
        print('load model from ', save_path)
        
        checkpoint = torch.load(save_path)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self.net_flow.load_state_dict(weights)

    def get_current_scalars(self):
        losses = {}
        losses['loss_G'] = self.loss_G.item()
        losses['loss_L1'] = self.loss_L1.item()
        losses['loss_percep'] = self.loss_perceptual.item()
        losses['loss_warp'] = self.loss_warp.item()
        losses['loss_flow_smooth'] = self.loss_flow_smoothness.item()
        return losses

    def get_current_visuals(self):
        output_visuals = {}

        output_visuals['im_rs'] = self.im_rs[:,-3:,:,:].clone()
        output_visuals['im_gs'] = self.im_gs_f[:,-3:,:,:].clone()
        
        return output_visuals
