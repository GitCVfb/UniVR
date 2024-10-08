import sys
sys.path.append('RIFE')
from IFNet_m import *

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float().cuda() 
    y = torch.arange(0, H, 1).float().cuda()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    
    grid = torch.stack([xx, yy], dim=0) 
    return grid

def GS_temporal_offset(t, grid_rows, H, gamma=1.0):# 0 <= t <= 1,  -gamma/2 <= t <= 1 - gamma/2
    tau = t + gamma - gamma * grid_rows/H + 0.0001
    
    return tau

def interflow_syn(bidir_optical_flow, t, gamma=1.0):# -gamma/2 <= tau <= 1 - gamma/2
    _, _, H, W = bidir_optical_flow.size()
    
    grid_rows = generate_2D_grid(H, W)[1]
    grid_rows = grid_rows.unsqueeze(0).unsqueeze(0)
    
    tau_y0_forward_y1_backward = GS_temporal_offset(t, grid_rows, H, gamma)
    time_distance = tau_y0_forward_y1_backward

    y1_forward = grid_rows + bidir_optical_flow[:,1]
    tau_y1_forward = GS_temporal_offset(t, y1_forward, H, gamma)

    y0_backward = grid_rows + bidir_optical_flow[:,3]
    tau_y0_backward = GS_temporal_offset(t, y0_backward, H, gamma)
    '''
    denom_forward = 1.0 + gamma * (tau_y1_forward - tau_y0_forward_y1_backward) / H
    denom_backward = 1.0 + gamma * (tau_y0_forward_y1_backward - tau_y0_backward) / H
    '''
    denom_forward = 1.0 + gamma * (bidir_optical_flow[:,1].unsqueeze(1)) / H
    denom_backward = 1.0 + gamma * (-bidir_optical_flow[:,3].unsqueeze(1)) / H

    factor_left_forward = tau_y0_forward_y1_backward / denom_forward
    factor_right_forward = (1.0 - tau_y1_forward) / denom_forward

    factor_left_backward = tau_y0_backward / denom_backward
    factor_right_backward = (1.0 - tau_y0_forward_y1_backward) / denom_backward

    flow_t_0_from_forward = -1.0 * factor_left_forward * bidir_optical_flow[:,0:2]
    flow_t_0_from_backward = factor_left_backward * bidir_optical_flow[:,2:4]

    flow_t_1_from_forward = factor_right_forward * bidir_optical_flow[:,0:2]
    flow_t_1_from_backward = -1.0 * factor_right_backward * bidir_optical_flow[:,2:4]

    flow_t_0_syn = (1.0 - time_distance) * flow_t_0_from_forward + time_distance * flow_t_0_from_backward
    flow_t_1_syn = (1.0 - time_distance) * flow_t_1_from_forward + time_distance * flow_t_1_from_backward

    return torch.cat((flow_t_0_syn, flow_t_1_syn), 1)


class UniVR(nn.Module):
    def __init__(self):
        super(UniVR, self).__init__()

        self.UVR = IFNet_m()
    
    def forward(self, img, timestep=0.0, gamma=1.0, scale=1, scale_list=[4, 2, 1], TTA=False):
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
     
        _, _, H, W = img.size()
        grid_rows = generate_2D_grid(H, W)[1]
        grid_rows = grid_rows.unsqueeze(0).unsqueeze(0)
        time_offset = GS_temporal_offset(timestep, grid_rows, H, gamma)
        time_offset = (img[:, :1].clone() * 0 + 1) * time_offset

        #time_offset = (img[:, :1].clone() * 0 + 1) * timestep #used to directly test VFI that produces a middle RS image

        
        flow, mask, merged = self.UVR.forward(img, time_offset, scale_list)

        if TTA == False:
            return merged, flow, mask
        else:
            flow2, mask2, merged2 = self.UVR(img.flip(2).flip(3), time_offset, scale_list)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2, flow, mask

