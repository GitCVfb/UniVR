import sys
sys.path.append('RIFE')
from warplayer import warp

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
        self.model_names=['flow']
        self.net_flow = UniVR().cuda()
        #self.net_flow = torch.nn.DataParallel(UniVR()).cuda()
        
        # load in initialized network parameters
        if opts.test_pretrained_VFI:
            self.load_pretrained_GS_model(opts.model_label, self.opts.log_dir)#load pretrained GS-based VFI model directly
        elif not opts.is_training or opts.continue_train:
            self.load_checkpoint(opts.model_label)
        else:
            self.load_pretrained_GS_model(self.opts.model_label_pretrained_GS, self.opts.log_dir_pretrained_GS)#load pretrained GS-based VFI model
        
        if self.opts.is_training:
            # initialize optimizers
            
            self.optimizer_G = torch.optim.Adam([{'params': self.net_flow.parameters()}], lr=opts.lr)

            self.optimizer_names = ['G']
            self.build_lr_scheduler()
            
            # create losses
            self.loss_fn_perceptual = PerceptualLoss(loss=nn.L1Loss())
            self.loss_fn_L1 = L1Loss()
            self.loss_fn_tv2 = VariationLoss(nc=2)

            self.downsampleX2 = nn.AvgPool2d(2, stride=2)
            
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

    def forward(self, time, gamma):# -gamma/2 <= time <= 1 - gamma/2
        
        gs_t_final, flow, mask = self.net_flow(self.im_rs, time, gamma)
        
        if self.opts.is_training:
            return gs_t_final, flow, mask
        
        return gs_t_final[2]


    def optimize_parameters(self):
        
        #===========================================================#
        #                   Initialize losses                       #
        #===========================================================#
        self.loss_L1 = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_warping = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_perceptual = torch.tensor([0.], requires_grad=True).cuda().float()
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
            
            pred_im, flow, mask = self.forward(timesteps[tt], self.opts.gamma)
            #===========================================================#
            #                       Compute losses                      #
            #===========================================================#
            gs_gt = im_gt.clone()
            rs_gt = self.im_rs.clone()

            weights_level = [0.25, 0.5, 1.0]
            
            num_sup = 3 #3
            for i in range(3):
                self.loss_L1 += weights_level[i] * self.charbonier_loss(pred_im[i], im_gt, mean=True) * self.opts.lamda_L1 *10.0
                self.loss_perceptual += weights_level[i] * self.loss_fn_perceptual.get_loss(pred_im[i], im_gt) * self.opts.lamda_perceptual
                self.loss_flow_smoothness += self.opts.lamda_flow_smoothness * (self.loss_fn_tv2(flow[i][:, :2], mean=True) + self.loss_fn_tv2(flow[i][:, 2:4], mean=True)) / 2.0   

        self.loss_perceptual = self.loss_perceptual / (len(timesteps)*num_sup)
        self.loss_L1 = self.loss_L1 / (len(timesteps)*num_sup)
        self.loss_flow_smoothness = self.loss_flow_smoothness / (len(timesteps)*num_sup)

        # sum them up
        self.loss_G = self.loss_L1 +\
                        self.loss_perceptual +\
                        self.loss_flow_smoothness 
                        
        # Optimize 
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()
    
    def charbonier_loss(self, pred_im, im_gt, epsilon=0.001, mean=True):
        x = pred_im - im_gt
        loss = torch.mean(torch.sqrt(x ** 2 + epsilon ** 2))
        return loss
    
    # save networks to file 
    def save_checkpoint(self, label):
        self.save_network(self.net_flow, 'flow', label, self.opts.log_dir)

    def load_checkpoint(self, label):
        self.load_network(self.net_flow, 'flow', label, self.opts.log_dir)

    def load_pretrained_GS_model(self, label, save_dir):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        
        def convert2(param):
            return {
            k.replace("module.", "UVR."): v
                for k, v in param.items()
                if "module." in k
            }
        
        save_filename = '%s_net_%s.pkl' % (label, 'flow')
        save_path = os.path.join(save_dir, save_filename)
        print('load model from ', save_path)
        
        #self.net_flow.load_state_dict(convert(torch.load('{}/pre_net_flow.pkl'.format(save_dir))))
        #self.net_flow.load_state_dict(torch.load(save_path), False)
        self.net_flow.load_state_dict((convert2(torch.load(save_path))))
        
    
    def get_current_scalars(self):
        losses = {}
        losses['loss_G'] = self.loss_G.item()
        losses['loss_L1'] = self.loss_L1.item()
        losses['loss_percep'] = self.loss_perceptual.item()
        #losses['loss_warp'] = self.loss_warping.item()
        losses['loss_flow_smooth'] = self.loss_flow_smoothness.item()
        return losses

    def get_current_visuals(self):
        output_visuals = {}

        output_visuals['im_rs'] = self.im_rs[:,-3:,:,:].clone()
        output_visuals['im_gs'] = self.im_gs_f[:,-3:,:,:].clone()
        
        return output_visuals
