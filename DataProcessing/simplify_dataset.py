import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)

def simplify_grasp_labels(root, save_path):
    obj_names = list(range(88))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in obj_names:
        print('\nsimplifying object {}:'.format(i))
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        points = label['points']
        scores = label['scores']
        offsets = label['offsets']
        width = offsets[:, :, :, :, 2]
        np.savez(os.path.join(save_path, '{}_labels.npz'.format(str(i).zfill(3))),
                 points=points, scores=scores, width=width)


if __name__ == '__main__':
    cfgs = parser.parse_args()
    root = cfgs.dataset_root
    save_path = os.path.join(root, 'grasp_label_simplified')
    simplify_grasp_labels(root, save_path)

