#!/bin/sh
#JSUB -q normal
#JSUB -n 12
#JSUB -m gpu02
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J task1

python generate_scenes.py --dataset_root /home/guihaiyuan/data/Benchmark/GraspNet_1Billion --camera_type realsense
  
