import os
import torch
import random
import argparse
import numpy as np
from skimage import io
import shutil
import imageio

import cv2
import flow_viz

from package_core.generic_train_test import *
from dataloader import *
from model_UniVR import *
from frame_utils import *

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser=argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')
parser.add_argument('--continue_train', type=bool, default=True, help='flags used to indicate if train model from previous trained weight')
parser.add_argument('--is_training', type=bool, default=False, help='flag used for selecting training mode or evaluation mode')
parser.add_argument('--seq_len', type=int, default=2)
parser.add_argument('--shuffle_data', type=bool, default=False)
parser.add_argument('--crop_sz_H', type=int, default=448, help='cropped image size height')
parser.add_argument('--crop_sz_W', type=int, default=640, help='cropped image size width')

parser.add_argument('--timestep', type=float, default=0.0, help='time of latent GS')

parser.add_argument('--model_label', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--results_dir', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)

parser.add_argument('--test_pretrained_VFI', type=bool, default=False)#whethre test pretrained GS-based VFI model
parser.add_argument('--gamma', type=float, default=0.45, help='readout time ratio')

parser.add_argument('--is_Fastec', type=int, default=0)

opts=parser.parse_args()

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelUniVR(opts)

##===================================================##
##**************** Train the network ****************##
##===================================================##
#Test and save all data in RS test set

class Demo(Generic_train_test):
    def test(self):
        with torch.no_grad():
            seq_lists = os.listdir(self.opts.data_dir)
            for seq in seq_lists:
                for i in range(0,49):
                    im_rs0_path = os.path.join(os.path.join(self.opts.data_dir, seq), 'RS/'+str(i).zfill(5)+'.png')
                    im_rs1_path = os.path.join(os.path.join(self.opts.data_dir, seq), 'RS/'+str(i+1).zfill(5)+'.png')
                    #print(im_rs0_path)

                    if not os.path.exists(im_rs0_path) or not os.path.exists(im_rs1_path):
                        continue

                    im_rs0 = torch.from_numpy(io.imread(im_rs0_path).transpose(2,0,1))[:3,:,:].unsqueeze(0).clone()
                    im_rs1 = torch.from_numpy(io.imread(im_rs1_path).transpose(2,0,1))[:3,:,:].unsqueeze(0).clone()

                    im_rs = torch.cat([im_rs0,im_rs1], dim=1).float()/255.
                    
                    _input = [im_rs, None, None, None]
                    B,C,H,W = im_rs.size()
                
                    self.model.set_input(_input)
                    #pred_gs_m_final = self.model.forward(self.opts.timestep, self.opts.gamma)
                    pred_gs_m_final = self.model.forward(1-self.opts.gamma/2, self.opts.gamma)
                    
                    # save results
                    save_path = os.path.join(self.opts.results_dir, seq)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    io.imsave(os.path.join(os.path.join(self.opts.results_dir, seq), str(i+1).zfill(5), '_pred_gs_m.png').replace('/_', '_'), (pred_gs_m_final.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                    print('saved', self.opts.results_dir, seq, i, 'pred_m.png')
                            
Demo(model, opts, None, None).test()


