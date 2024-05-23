import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata
import os
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scale_data(input):
    min_val = np.min(input)
    max_val = np.max(input)
    scaled = 2 * (input - min_val) / (max_val - min_val) - 1
    return scaled

def gen_train_data(batchsize, seq_len, dim=1, com_num=20):
    ### generated data has shape [batchsize, seq_len, dim*2], each one in the batch is a distribution
    data = np.zeros((batchsize, seq_len, dim*2))
    for i in range(batchsize):
        num_components = np.random.randint(1, com_num+1)
        
        weights = np.random.dirichlet(np.ones(num_components))
        
        means = [np.random.uniform(-5, 5, size=2*dim) for _ in range(num_components)]
        
        covs = []
        for j in range(num_components):
            a11 = np.sum(np.random.uniform(-3, 3, size=(2*dim))**2)
            a22 = np.sum(np.random.uniform(-3, 3, size=(2*dim))**2)
            a12 = np.random.uniform(-np.sqrt(a11*a22), np.sqrt(a11*a22))
            cov = np.array([[a11, a12],[a12, a22]]) + 0.01*np.eye(2*dim)
            covs.append(cov)

        gm = GaussianMixture(n_components=num_components)
        gm.weights_ = np.array(weights)
        gm.means_ = np.array(means)
        gm.covariances_ = np.array(covs)
        #gm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs)).transpose((0, 2, 1))

        joint_samples, labels = gm.sample(n_samples=seq_len)
        
        joint_samples = np.array(joint_samples[:, [0,1]])
        joint_samples[:, 0] = rankdata(joint_samples[:, 0])/seq_len
        joint_samples[:, 1] = rankdata(joint_samples[:, 1])/seq_len
        #print(np.max(joint_samples[:,0]), np.max(joint_samples[:,0]))

        data[i] = np.array(joint_samples)
    return data

