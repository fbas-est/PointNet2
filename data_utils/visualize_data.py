import open3d as o3d
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.getcwd())
from dataloaders.modelnet_dataloader import ModelNetDataloader



parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='data', help='root')
parser.add_argument('--dataset_name', type=str, default='modelnet40_normal_resampled', help='dataset name')
parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')


if __name__ == '__main__':
    args = parser.parse_args()

    dataset_path = os.path.join(args.root, args.dataset_name)

    if args.dataset_name == 'modelnet40_normal_resampled':
        dataset = ModelNetDataloader(dataset_path, use_normals=args.use_normals, num_class=40)

    # Get 10 random indeces and visualize the pointset
    for i in range(10):
        random_idx = np.random.randint(0, len(dataset))
        point_set, label, class_name = dataset[random_idx]

        point_set = point_set.transpose(1,0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_set[:, 0:3])
        if args.use_normals:
            pcd.normals = o3d.utility.Vector3dVector(point_set[:, 3:6])
        
        window_name = class_name.upper()
        o3d.visualization.draw_geometries([pcd], window_name = window_name)