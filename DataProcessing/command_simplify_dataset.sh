#!/bin/sh
#JSUB -q normal
#JSUB -n 16
#JSUB -m gpu01
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J task2

python simplify_dataset.py --dataset_root /home/guihaiyuan/data/Benchmark/GraspNet_1Billion
