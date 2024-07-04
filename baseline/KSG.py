import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
from scipy.spatial import KDTree
import scipy.special
from sklearn.mixture import GaussianMixture
import time
import scipy.spatial as ss
import scipy.stats as sst
import os
import pickle
import matplotlib.pyplot as plt

def scale_data(input_tensor):
    min_val = np.min(input_tensor)
    max_val = np.max(input_tensor)
    scaled_tensor = 2 * (input_tensor - min_val) / (max_val - min_val) - 1
    return scaled_tensor

def kraskov_mi(x, y, k=5):

    # x,y has shape [N,d]
    N = x.shape[0]
    dx = x.shape[1]   	
    dy = y.shape[1]
    data = np.concatenate((x,y),axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
    ans_xy = -digamma(k) + digamma(N) + (dx+dy)*np.log(2)#2*log(N-1) - digamma(N) #+ vd(dx) + vd(dy) - vd(dx+dy)
    ans_x = digamma(N) + dx*np.log(2)
    ans_y = digamma(N) + dy*np.log(2)
    for i in range(N):
        ans_xy += (dx+dy)*np.log(knn_dis[i])/N
        ans_x += -digamma(len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dx*np.log(knn_dis[i])/N
        ans_y += -digamma(len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dy*np.log(knn_dis[i])/N
            
    return ans_x+ans_y-ans_xy

if __name__ == '__main__':

    rou = 0.5
    x, y = np.random.multivariate_normal(mean=[0,0], cov=[[1,rou],[rou,1]], size=2000).T
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    result = kraskov_mi(x, y, k=5)
    real_MI = -np.log(1-rou**2)/2