
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S], S are the number of point to index.
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1) # Keep the batches by creating a list with ones in every other dim
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1 # Keep the number of sampled points
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(source_points, target_points):
    """
    Calculate squared Euclidean distance between source and target points.
    
    Args:
    - source_points (torch.Tensor): Source points tensor of shape [B, N, C].
    - target_points (torch.Tensor): Target points tensor of shape [B, M, C].
    
    Returns:
    - distances (torch.Tensor): Per-point squared distance tensor of shape [B, N, M].
    """
    # Extract shapes
    B, N, C = source_points.size()
    _, M, _ = target_points.size()
    
    # Reshape source and target points for broadcasting
    source_points_expanded = source_points.unsqueeze(2)  # [B, N, 1, C]
    target_points_expanded = target_points.unsqueeze(1)  # [B, 1, M, C]
    
    # Compute squared Euclidean distance
    differences = source_points_expanded - target_points_expanded  # [B, N, M, C]
    squared_distances = torch.sum(differences**2, dim=-1)  # [B, N, M]
    
    return squared_distances

    
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz) # creates a matrix of [B, M, N] with all the distances
    group_idx[sqrdists > radius ** 2] = N # Calculates if the point is within the local region or not.
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # sort the points based on the N dimension and it keeps only the sampled number of those that holds the minimum distance
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask] # Replaces the outbound points with the index of the closest neighbor therefore the same points.
    return group_idx


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

if __name__ == '__main__':
    B, N, C, S = 2, 100, 3, 30
    points = torch.randn(B, N, C) # Random points data
    idx = torch.randint(0, N, (B, S))  # Random sample indices between 0 and N
    print('Original Pointcloud:', points.size(), 'Random set of indeces:', idx.size())
    
    # Test index_points
    new_points = index_points(points, idx)
    print('Sampled Points:', new_points.size())
    
    # Test FPS
    sampled_points = farthest_point_sample(points, 10)
    print("FPS:", sampled_points.shape)

    # Test query_ball_point
    group_idx = query_ball_point(0.5, 5, points, new_points)
