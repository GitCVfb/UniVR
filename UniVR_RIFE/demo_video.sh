#!/bin/bash

# create an empty folder for experimental results
mkdir -p experiments/results_demo_carla_video
mkdir -p experiments/results_demo_faster_video
mkdir -p experiments/results_demo_bsrsc_all

cd deep_unroll_net


python inference_demo_video.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_carla_video \
            --data_dir='../demo_video/Carla' \
            --crop_sz_H=448 \
            --gamma=1.0 \
            --log_dir=../deep_unroll_weights/pre_carla_ft

:<<!
python inference_demo_video.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_faster_video \
            --data_dir='../demo_video/Fastec' \
            --crop_sz_H=480 \
            --gamma=1.0 \
            --log_dir=../deep_unroll_weights/pre_fastec_ft
!


:<<!
python inference_demo_bsrsc_all.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_bsrsc_all \
            --data_dir=/data/local_userdata/fanbin/raw_data/BSRSC/test \
            --crop_sz_H=768 \
            --gamma=0.45 \
            --log_dir=../deep_unroll_weights/pre_bsrsc_ft
!