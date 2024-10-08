import sys
sys.path.append('superslomo')
from model_vfi import *

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float().cuda() 
    y = torch.arange(0, H, 1).float().cuda()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    
    grid = torch.stack([xx, yy], dim=0) 
    return grid

def GS_temporal_offset(t, grid_rows, H, gamma=1.0):# 0 <= t <= 1,  -gamma/2 <= t <= 1 - gamma/2
    tau = t + gamma - gamma * grid_rows/H + 0.0001
    #tau = t + gamma - gamma * grid_rows/(H-1) + 0.0001
    
    return tau

def interflow_syn(bidir_optical_flow, t, gamma=1.0):# -gamma/2 <= tau <= 1 - gamma/2
    _, _, H, W = bidir_optical_flow.size()
    
    grid_rows = generate_2D_grid(H, W)[1]
    grid_rows = grid_rows.unsqueeze(0).unsqueeze(0)
    grid_rows = (bidir_optical_flow[:, :1].clone() * 0 + 1) * grid_rows
    
    tau_y0_forward_y1_backward = GS_temporal_offset(t, grid_rows, H, gamma)
        
    time_distance = tau_y0_forward_y1_backward.clone()
    time_distance = torch.clamp(time_distance, 0, 1)
    time_distance = (1 - 2e-2) * time_distance + 1e-2

    y1_forward = grid_rows + bidir_optical_flow[:,1].unsqueeze(1)
    tau_y1_forward = GS_temporal_offset(t, y1_forward, H, gamma)

    y0_backward = grid_rows + bidir_optical_flow[:,3].unsqueeze(1)
    tau_y0_backward = GS_temporal_offset(t, y0_backward, H, gamma)
    '''
    denom_forward = 1.0 + gamma * (tau_y1_forward - tau_y0_forward_y1_backward) / H
    denom_backward = 1.0 + gamma * (tau_y0_forward_y1_backward - tau_y0_backward) / H
    '''
    denom_forward = 1.0 + gamma * (bidir_optical_flow[:,1].unsqueeze(1)) / H
    denom_backward = 1.0 + gamma * (-bidir_optical_flow[:,3].unsqueeze(1)) / H

    factor_left_forward = tau_y0_forward_y1_backward / denom_forward
    factor_right_forward = (1.0 - tau_y1_forward) / denom_forward  # = 1.0-factor_left_forward

    factor_left_backward = tau_y0_backward / denom_backward
    factor_right_backward = (1.0 - tau_y0_forward_y1_backward) / denom_backward  # = 1.0-factor_left_backward

    flow_t_0_from_forward = - factor_left_forward * bidir_optical_flow[:,0:2]
    flow_t_0_from_backward = factor_left_backward * bidir_optical_flow[:,2:4]

    flow_t_1_from_forward = factor_right_forward * bidir_optical_flow[:,0:2]
    flow_t_1_from_backward = - factor_right_backward * bidir_optical_flow[:,2:4]

    flow_t_0_syn = (1.0 - time_distance) * flow_t_0_from_forward + time_distance * flow_t_0_from_backward
    flow_t_1_syn = (1.0 - time_distance) * flow_t_1_from_forward + time_distance * flow_t_1_from_backward

    return flow_t_0_syn, flow_t_1_syn


class UniVR(nn.Module):
    def __init__(self):
        super(UniVR, self).__init__()

        self.UVR = UNet(20, 5)

    def forward(self, img, bidir_optical_flow, timestep=0.0, gamma=1.0): # 0 <= gamma <= 1,  -gamma/2 <= timestep <= 1 - gamma/2
        I0 = img[:,0:3,:,:].clone()
        I1 = img[:,3:6,:,:].clone()
        B,C,H,W = I0.size()
        
        flowBackWarp = backWarp(W, H, device)
        flowBackWarp = flowBackWarp.to(device)
        
        F_0_1 = bidir_optical_flow[:B,]
        F_1_0 = bidir_optical_flow[B:,]

        optical_flow_bidir = torch.cat((F_0_1, F_1_0), 1)

        F_t_0, F_t_1 = interflow_syn(optical_flow_bidir, timestep, gamma)

        g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
        g_I1_F_t_1 = flowBackWarp(I1, F_t_1)
        
        intrpOut = self.UVR(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
        
        F_t_0_f = intrpOut[:, 0:2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1   = 1 - V_t_0
        
        g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
        g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

        grid_rows = generate_2D_grid(H, W)[1]
        grid_rows = grid_rows.unsqueeze(0).unsqueeze(0)
        time_distance = GS_temporal_offset(timestep, grid_rows, H, gamma)
        time_distance = torch.clamp(time_distance, 0, 1)
        time_offset = (1 - 2e-2) * time_distance + 1e-2
    
        time_offset_right = (img[:, :1].clone() * 0 + 1) * time_offset
        time_offset_left = 1.0 - time_offset_right
        
        Ft_p = (time_offset_left * V_t_0 * g_I0_F_t_0_f + time_offset_right * V_t_1 * g_I1_F_t_1_f) / (time_offset_left * V_t_0 + time_offset_right * V_t_1)

        flow_list = [F_t_0_f, F_t_1_f, F_t_0, F_t_1, F_0_1, F_1_0]
        imgs_list = [g_I0_F_t_0_f, g_I1_F_t_1_f, g_I0_F_t_0, g_I1_F_t_1]

        return torch.clamp(Ft_p, 0, 1), imgs_list, flow_list


