#!/bin/sh
#JSUB -q normal
#JSUB -n 2
#JSUB -m gpu01
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J task1


CUDA_VISIBLE_DEVICES=0 python generate_graspness.py --dataset_root /hpcfiles/users/guihaiyuan/data/Benchmark/graspnet --camera_type kinect --idx 90
# python simplify_dataset.py --dataset_root /hpcfiles/users/guihaiyuan/data/Benchmark/graspnet
# python vis_graspness.py
# CUDA_VISIBLE_DEVICES=1 python test_obs.py --dump_dir logs/dump_sbg_rs_4_obs_CD0  --checkpoint_path logs/log_sbg_rs_4/checkpoint.tar --seg_checkpoint_path logs/log_insseg/checkpoint.tar --collision_thresh 0 --camera realsense
# torchrun  --standalone --nnodes=1 --nproc_per_node=2 train.py
  
