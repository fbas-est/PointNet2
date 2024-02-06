import os
from torch.utils.data import Dataset
import numpy as np
import torch
import time
import warnings

warnings.filterwarnings('ignore')

def random_sample(points, samples):
    if samples > points.shape[0]:
        raise ValueError("Number of points to sample is greater than the size of the point set.")

    np.random.shuffle(points)
    sampled_points = points[:samples]
    return sampled_points


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
        centroid = points[farthest].reshape(1, 6)
        dist = np.sum((points - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=-1)
    points = points[centroids.astype(np.int32)]
    return points


class ModelNetDataloader(Dataset):

    def __init__(self, root, split='train', num_class = 10, use_normals = True, num_points = None, fast_sample=True):
        self.root = root
        self.split = split
        self.num_class = num_class
        self.use_normals = use_normals
        self.num_points = num_points
        self.fast_sample = fast_sample
        class_names_path = os.path.join(root, f'modelnet{num_class}_shape_names.txt')

        self.class_names = [line.strip() for line in open(class_names_path)]
        self.classes = dict(zip(self.class_names, range(len(self.class_names))))

        if split == 'train':
            train_data_path = os.path.join(root, f'modelnet{num_class}_train.txt')
            self.class_ids = [line.strip() for line in open(train_data_path)]
        elif split == 'test':
            test_data_path = os.path.join(root, f'modelnet{num_class}_test.txt')
            self.class_ids = [line.strip() for line in open(test_data_path)]

        self.process_data_paths = []
        for i in range(len(self.class_ids)):
            folder_name = self.class_ids[i].rsplit('_', 1)[0]
            self.process_data_paths.append(os.path.join(root, folder_name, self.class_ids[i]+'.txt'))


    def __len__(self):
        return len(self.process_data_paths)
    
    def get_item(self, idx):
        data_path = self.process_data_paths[idx]

        # load points and normalize the values
        point_set = np.loadtxt(data_path, delimiter=',').astype(np.float32)

        if self.num_points!=None:
            if not self.fast_sample:
                point_set = farthest_point_sample(point_set, self.num_points)
            else:
                point_set = random_sample(point_set, self.num_points)

        mean_pts = np.mean(point_set[:, 0:3])
        centered_pts = point_set[:, 0:3] - mean_pts
        max_abs = np.max(np.abs(centered_pts), axis=0)
        point_set[:, 0:3] = centered_pts*1.0 / max_abs


        # check if normals will be used, if yes then normalize their values
        if self.use_normals == False:
            point_set = point_set[:,0:3]
        else:
            mean_normals = np.mean(point_set[:, 3:6])
            centered_normals = point_set[:, 3:6] - mean_normals
            max_abs = np.max(np.abs(centered_normals), axis=0)
            point_set[:, 3:6] = centered_normals*1.0 / max_abs

        # Find the corresponding class name
        cls_name = self.class_ids[idx].rsplit('_', 1)[0]
 
        # Find the corresponding class label
        label = self.classes[cls_name]
        
        point_set = point_set.transpose(1, 0)

        return point_set, int(label), cls_name

    def __getitem__ (self, idx):
        return self.get_item(idx)


if __name__ == '__main__':
    root = 'data/modelnet40_normal_resampled'
    dataset = ModelNetDataloader(root = root, num_class=40)
    length = len(dataset)
    print(dataset[9000])

