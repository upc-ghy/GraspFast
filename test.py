import os
import sys
import numpy as np
import argparse
import time

import torch
from torch.utils.data import DataLoader
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'GraspFastModel'))
sys.path.append(os.path.join(ROOT_DIR, 'DataProcessing'))

import torch.nn.functional as F
from dsn import DSN,cluster
from graspfast import GraspFast, pred_decode
from graspfast_dataset import GraspFastDataset, minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='/home/guihaiyuan/data/Benchmark/graspnet')
parser.add_argument('--checkpoint_path', required=True)
parser.add_argument('--seg_checkpoint_path', required=True)
parser.add_argument('--dump_dir', required=True)
parser.add_argument('--seed_feat_dim', default=512, type=int)
parser.add_argument('--camera', required=True)
parser.add_argument('--num_point', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--voxel_size', type=float, default=0.005)
parser.add_argument('--collision_thresh', type=float, default=0.01)
parser.add_argument('--voxel_size_cd', type=float, default=0.01)
parser.add_argument('--num_workers', type=int, default=15)
cfgs = parser.parse_args()

if not os.path.exists(cfgs.dump_dir): os.mkdir(cfgs.dump_dir)
 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

TEST_DATASET = GraspFastDataset(cfgs.dataset_path, split='test', camera=cfgs.camera, num_points=cfgs.num_point, voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False, load_label=False)

print(len(TEST_DATASET))
SCENE_LIST = TEST_DATASET.scene_list()
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False, num_workers=cfgs.num_workers, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)

net = GraspFast(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

checkpoint = torch.load(cfgs.checkpoint_path, map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))


seg_net = DSN(input_feature_dim=0)
seg_net.to(device)

checkpoint = torch.load(cfgs.seg_checkpoint_path)
seg_net.load_state_dict(checkpoint['model_state_dict'])

def inference():
    batch_interval = 100
    stat_dict = {}
    net.eval()
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
        
        with torch.no_grad():
            end_points = seg_net(batch_data)
            batch_xyz_img = end_points["point_clouds"]
            B, _, N = batch_xyz_img.shape
            batch_offsets = end_points["center_offsets"]
            batch_fg = end_points["foreground_logits"]
            batch_fg = F.softmax(batch_fg, dim=1)
            batch_fg = torch.argmax(batch_fg, dim=1)
            clustered_imgs = []
            for i in range(B):
                clustered_img, uniq_cluster_centers = cluster(batch_xyz_img[i], batch_offsets[i].permute(1, 0),
                                                              batch_fg[i])
                clustered_img = clustered_img.unsqueeze(0)
                clustered_imgs.append(clustered_img)
            end_points['seed_cluster'] = torch.cat(clustered_imgs,dim=0)
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)

            if cfgs.collision_thresh > 0:
                cloud = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]

            save_dir = os.path.join(cfgs.dump_dir, SCENE_LIST[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, str(data_idx%256).zfill(4)+'.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

def evaluate():
    ge = GraspNetEval(root=cfgs.dataset_path, camera=cfgs.camera, split='test')
    res, ap = ge.eval_all(cfgs.dump_dir, proc=cfgs.num_workers)
    save_dir = os.path.join(cfgs.dump_dir, 'ap_{}.npy'.format(cfgs.camera))
    np.save(save_dir, res)

if __name__=='__main__':
    inference()
    # evaluate()
