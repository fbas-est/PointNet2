import open3d as o3d
import numpy as np
import torch
import argparse
from models.pointnet2_cls import PointNet2Classification
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='test_data', help='Folder containing the test data')  
    parser.add_argument('--model_path', type=str, default='run/modelnet40_normal_resampled/pointnet2_cls/run_0/last.pth.tar', help='Folder containing the trained model')  
    parser.add_argument('--class_names_path', type=str, default='data/modelnet40_normal_resampled/modelnet40_shape_names.txt', help='Folder containing the class names')  
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--number_classes', type=int, default=40, help='Number of training classes')
    parser.add_argument('--vis_pcd', action='store_true', default=True, help='Visualize pcds for each prediction')
    args = parser.parse_args()

    # Define operation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate and load trained model
    model = PointNet2Classification(num_class = args.number_classes, normal_channel=args.use_normals)
    if args.model_path != None:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # Make a list with all the paths
    file_paths = [os.path.join(args.data_path,  file_name) for file_name in os.listdir(args.data_path)]
    
    # Make a list with the valid file extentions
    valid_ext = ['ply', 'xyz', 'txt', 'pcd', 'pts']
    # Load all the pcds into a list
    pcd_list = []
    for path in file_paths:
        if path.rsplit('.')[1] in valid_ext:
            pcd_list.append(o3d.io.read_point_cloud(path))

    # Read the class mapping from the text file
    class_mapping_file = args.class_names_path
    class_mapping = {}

    with open(class_mapping_file, 'r') as file:
        for i, line in enumerate(file):
            parts = line.strip().split()  # Split each line into parts
            class_mapping[i] =  parts[0]


    for pcd in pcd_list:
        # Convert points and normals to NumPy arrays
        points = np.asarray(pcd.points)  
        if args.use_normals:
            normals = np.asarray(pcd.normals)
            points = np.concatenate((points, normals), axis=-1)  

        # Normalize set of points and normals
        mean_pts = np.mean(points[:, 0:])
        centered_pts = points[:, 0:] - mean_pts
        max_abs = np.max(np.abs(centered_pts), axis=0)
        points[:, 0:] = centered_pts*1.0 / max_abs

        # Prepare points for the trained model
        points = points.transpose(1, 0)
        points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)

        # Generate result
        pred, pts = model(points_tensor)
        pred_np = pred.detach().cpu().numpy()


        print('Detected Class::', class_mapping[np.argmax(pred_np)].upper())
        if args.vis_pcd:
            o3d.visualization.draw_geometries([pcd])

