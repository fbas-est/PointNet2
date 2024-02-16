import open3d as o3d
import numpy as np
import torch
import argparse
from models.pointnet2_cls import PointNet2Classification
import os


def farthest_point_sample(points, samples):
    """
    Input:
        points: input points data, [N, 3]
        samples: are the number of point to index.
    Return:
        new_points:, indexed points data, [samples, 3]
    """
    N, C = points.shape
    centroids = np.zeros(samples, dtype=np.int64)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N, dtype=np.int64)
    new_points = np.zeros([samples, C])
    for i in range(samples):
        centroids[i] = farthest
        centroid = points[farthest].reshape(1, 3)
        dist = np.sum((points - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=-1)
    points = points[centroids.astype(np.int32)]
    return points

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='test_data', help='Folder containing the test data')  
    parser.add_argument('--model_path', type=str, default='run/modelnet40_normal_resampled/pointnet2_cls/run_1/last.pth.tar', help='Folder containing the trained model')  
    parser.add_argument('--class_names_path', type=str, default='data/modelnet40_normal_resampled/modelnet40_shape_names.txt', help='Folder containing the class names')  
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--number_classes', type=int, default=40, help='Number of training classes')
    parser.add_argument('--vis_pcd', action='store_true', default=True, help='Visualize pcds for each prediction')
    parser.add_argument('--num_points', type=int, default=None, help='Point Number')
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
            if path.rsplit('.')[1] != 'txt':
                curr_pcd = o3d.io.read_point_cloud(path)
                points = np.concatenate((np.asarray(curr_pcd.points), np.asarray(curr_pcd.normals)), axis=-1)
                pcd_list.append(points)
            else:
                pcd_list.append(np.loadtxt(path, delimiter=',').astype(np.float32))

    # Read the class mapping from the text file
    class_mapping_file = args.class_names_path
    class_mapping = {}

    with open(class_mapping_file, 'r') as file:
        for i, line in enumerate(file):
            parts = line.strip().split()  # Split each line into parts
            class_mapping[i] =  parts[0]


    for pcd in pcd_list:
        # Convert points and normals to NumPy arrays
        points = pcd
        if not args.use_normals:
            points = points[:,:3]

        points = farthest_point_sample(points, args.num_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

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
            o3d.visualization.draw_geometries([pcd], width = 1280, height = 720)
            

