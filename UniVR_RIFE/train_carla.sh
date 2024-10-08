#!/bin/bash

# !! Updata the path to the dataset and directory to 
# !! save your trained models with your own local path !!
carla_dataset_type=Carla
carla_root_path_training_data=/data/local_userdata/fanbin/raw_data/carla/data_train/train/

log_dir_pretrained_GS=/home/fanbin/fan/UniVR/UniVR_RIFE/deep_unroll_weights/Pretrained/
log_dir=/home/fanbin/fan/UniVR/UniVR_RIFE/deep_unroll_weights/
#
cd deep_unroll_net


python train_UniVR.py \
          --dataset_type=$carla_dataset_type \
          --dataset_root_dir=$carla_root_path_training_data \
          --log_dir_pretrained_GS=$log_dir_pretrained_GS \
          --log_dir=$log_dir \
          --gamma=1.0 \
          --lamda_L1=10 \
          --lamda_perceptual=1 \
          --lamda_flow_smoothness=0.1 \
          --crop_sz_H=448 \
          #--continue_train=True \
          #--start_epoch=201 \
          #--model_label=200
          