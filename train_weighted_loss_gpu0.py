import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(ROOT_DIR, 'GraspFastModel'))
sys.path.append(os.path.join(ROOT_DIR, 'DataProcessing'))

from GraspFastModel.graspfast import GraspFast
from GraspFastModel.loss import get_loss, get_weighted_loss
from DataProcessing.graspnet_dataset import GraspNetDataset, minkowski_collate_fn, load_grasp_labels

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='/home/guihaiyuan/data/Benchmark/GraspNet_1Billion')
parser.add_argument('--camera', default='realsense')
parser.add_argument('--checkpoint_path', default=None)
parser.add_argument('--model_name', type=str, default='GraspFast')
parser.add_argument('--log_dir', default='logs/log_graspfast')
parser.add_argument('--num_point', type=int, default=20000)
parser.add_argument('--feat_channel', default=512, type=int)
parser.add_argument('--voxel_size', type=float, default=0.003)
parser.add_argument('--max_epoch', type=int, default=18)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--resume', action='store_true', default=False, help='Whether to resume from checkpoint')
cfgs = parser.parse_args()
EPOCH_CNT = 0
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None and cfgs.resume else None

if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

grasp_labels = load_grasp_labels(cfgs.dataset_path)
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_path, grasp_labels=grasp_labels, camera=cfgs.camera, split='train',
                                num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                remove_outlier=False, augment=True, load_label=True)

print('train dataset length: ', len(TRAIN_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, num_workers=2, shuffle=True, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
print('train dataloader length: ', len(TRAIN_DATALOADER))

net = GraspNet(feat_channel=cfgs.feat_channel, is_training=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate)

start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

def get_current_lr(epoch):
    lr = cfgs.learning_rate
    lr = lr * (0.95 ** epoch)
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch():
    stat_dict = {}
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    batch_interval = 20
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        end_points = net(batch_data_label)
        # loss, end_points = get_loss(end_points)
        loss, end_points = get_weighted_loss(end_points)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % batch_interval == 0:
            for key in sorted(stat_dict.keys()):
                stat_dict[key] = 0


def train(start_epoch):
    global EPOCH_CNT
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        np.random.seed()
        train_one_epoch()
        save_dict = {'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict(), 'model_state_dict': net.state_dict()}
        torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model_name + '_epoch' + str(epoch + 1).zfill(2) + '.tar'))


if __name__ == '__main__':
    train(start_epoch)




