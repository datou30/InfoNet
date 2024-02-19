import numpy as np
import torch
import torch.nn.functional as F

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


## 3d version, supports vector as input
def lookup_value(matrix, vector):
    if isinstance(vector, np.ndarray):
        vector = torch.from_numpy(vector)


    dim = matrix.shape[0]
    num = vector.shape[0]

    values = torch.zeros(num, 1)

    indice_low = torch.floor(((vector + 1) * (dim - 1) / 2)).to(torch.int)
    indice_up = torch.ceil(((vector + 1) * (dim - 1) / 2)).to(torch.int)

    merge = torch.stack((indice_low, indice_up), dim=1)
    #merge = torch.transpose(merge, 0, 2) 
    print(merge)

    for i in range(num):
      x, y, z = torch.meshgrid(merge[i,:,0], merge[i,:,1], merge[i,:,2])
      indices = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
      indices = torch.clamp(indices, 0, dim - 1)
      values[i] = torch.mean(torch.tensor([matrix[indice] for indice in indices]))
    return values


# 2d version, it supports vector input
def lookup_value_2d(matrix, vector):
    if isinstance(vector, np.ndarray):
        vector = torch.from_numpy(vector)
    matrix = matrix.squeeze(0)
    vector = vector.squeeze(0)
    dim = matrix.shape[0]
    num = vector.shape[0]

    values = torch.zeros(num, 1)
    pos = ((vector + 1) * (dim - 1) / 2).squeeze(0)
    indice_low = torch.floor(((vector + 1) * (dim - 1) / 2)).to(torch.int)
    indice_up = torch.ceil(((vector + 1) * (dim - 1) / 2)).to(torch.int)

    merge = torch.stack((indice_low, indice_up), dim=1)
    merge = merge.squeeze(0).permute(1,0,2)

    for i in range(num):
      x, y = torch.meshgrid(merge[:,i,0], merge[:,i,1])
      indices = torch.stack([x.flatten(), y.flatten()], dim=1)
      indices = torch.clamp(indices, 0, dim - 1).long()
      distances = (torch.sum((indices - pos[i,:])**2, dim=1))
      weights = 1 / (distances+0.0001)
      values[i] = torch.sum( (matrix[indices[:,0],indices[:,1]]).clone()*weights/torch.sum(weights) )   
    return values

def lookup_value_average(matrix, vector):
    if isinstance(vector, np.ndarray):
        vector = torch.from_numpy(vector)
    matrix = matrix.squeeze(0)
    vector = vector.squeeze(0)
    dim = matrix.shape[0]
    num = vector.shape[0]

    values = torch.zeros(num, 1)
    pos = ((vector + 1) * (dim - 1) / 2).squeeze(0)
    indice_low = torch.floor(((vector + 1) * (dim - 1) / 2)).to(torch.int)
    indice_up = torch.ceil(((vector + 1) * (dim - 1) / 2)).to(torch.int)

    merge = torch.stack((indice_low, indice_up), dim=1)
    merge = merge.squeeze(0).permute(1,0,2)

    for i in range(num):
      x, y = torch.meshgrid(merge[:,i,0], merge[:,i,1])
      indices = torch.stack([x.flatten(), y.flatten()], dim=1)
      indices = torch.clamp(indices, 0, dim - 1).long()
      values[i] = torch.mean( (matrix[indices[:,0],indices[:,1]]).clone() )   
    return values

def lookup_value_average_vec(matrix, vector):
    if isinstance(vector, np.ndarray):
        vector = torch.from_numpy(vector)
    matrix = matrix.squeeze(0)
    vector = vector.squeeze(0)
    dim = matrix.shape[0]
    num = vector.shape[0]

    indice_low = torch.floor(((vector + 1) * (dim - 1) / 2)).to(torch.int)

    indices = indice_low.unsqueeze(1)
    offset = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).unsqueeze(0).cuda()
    indices = indices + offset

    row_indices = indices[:, :, 0]
    col_indices = indices[:, :, 1]

    values = torch.mean(matrix[row_indices, col_indices], dim=1)
    print(values.shape)
    return values

def lookup_value_close(matrix, vector):
    if isinstance(vector, np.ndarray):
        vector = torch.from_numpy(vector)
    
    matrix = matrix.squeeze()
    vector = vector.squeeze()
    dim = matrix.shape[0]
    num = vector.shape[0]

    values = torch.zeros(num, 1)
    pos = (vector + 1) * (dim - 1) / 2
    indices = torch.round(((vector + 1) * (dim - 1) / 2)).to(torch.int).long()
    values = matrix[indices[:, 0], indices[:, 1]]
     
    return values

def lookup_value_bilinear(matrix, vector):
    if isinstance(vector, np.ndarray):
        vector = torch.from_numpy(vector)
    matrix = matrix.squeeze(0)
    vector = vector.squeeze(0)
    dim = matrix.shape[0]
    num = vector.shape[0]

    values = torch.zeros(num, 1)
    pos = ((vector + 1) * (dim - 1) / 2).squeeze(0)
    indice_low = torch.floor(((vector + 1) * (dim - 1) / 2)).to(torch.int)
    indice_up = torch.ceil(((vector + 1) * (dim - 1) / 2)).to(torch.int)

    merge = torch.stack((indice_low, indice_up), dim=1)
    merge = merge.squeeze(0).permute(1,0,2)

    for i in range(num):
      x, y = torch.meshgrid(merge[:,i,0], merge[:,i,1])
      indices = torch.stack([x.flatten(), y.flatten()], dim=1)
      indices = torch.clamp(indices, 0, dim - 1).long()
      x_low = indices[0,0]
      y_low = indices[0,1]
      dx = (vector[i,0]+1) * (dim-1)/2 - x_low
      dy = (vector[i,1]+1) * (dim-1)/2 - y_low
      values[i] = (1 - dx) * (1 - dy) * matrix[x_low-1, y_low-1] + dx * (1 - dy) *matrix[x_low,y_low-1]  + (1 - dx) * dy * matrix[x_low-1,y_low] + dx * dy * matrix[x_low,y_low]
    
    return values

def lookup_value_grid(matrix, vector, mode):
    if isinstance(vector, np.ndarray):
        vector = torch.from_numpy(vector)
    #matrix = matrix.squeeze(0)
    #vector = vector.squeeze(0)
    batchsize = matrix.shape[0]
    dim = matrix.shape[1]
    num = vector.shape[1]
    matrix = matrix.view(batchsize, 1, dim, dim)
    vector = vector.view(batchsize, num, 1, 2)
    value = (F.grid_sample(matrix, vector, mode, padding_mode='border', align_corners=True)).squeeze()
    return value    

