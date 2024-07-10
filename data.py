import numpy as np
import torch
import torchsort
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
    return data.astype(np.float32)

def gen_train_data_weighted(batchsize, seq_len, max_num_components=10, dim=1, com_num=20):
    ### generated data has shape [batchsize, seq_len, dim*2], each one in the batch is a distribution
    data = np.zeros((batchsize, seq_len, dim*2))
    components_list = np.arange(1, max_num_components+1)
    #weights = np.arange(1, max_num_components+1)
    weights = np.array([i if i <= 10 else 5 for i in range(1, max_num_components + 1)])
    for i in range(batchsize):
        #num_components = np.random.randint(1, com_num+1)
        num_components = np.random.choice(components_list, p=weights/weights.sum())
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
    return data.astype(np.float32)

def gen_gauss_xyz(seq_len, dim=1):
    mean = np.random.uniform(-5, 5, size=3*dim)
    A = np.random.uniform(-3, 3, size=(3*dim, 3*dim))  # Generate a random matrix
    cov = np.dot(A, A.T) + 0.001*np.eye(3*dim)
    samples = np.random.multivariate_normal(mean, cov, seq_len).astype(np.float32)

    var_x = cov[0, 0]
    var_y = cov[1, 1]
    var_z = cov[2, 2]
    cov_xy = cov[0, 1]
    cov_yz = cov[1, 2]
    mi_xy = 0.5 * np.log(var_x * var_y / (var_x * var_y - cov_xy**2))
    mi_yz = 0.5 * np.log(var_y * var_z / (var_y * var_z - cov_yz**2))
    return samples, mi_xy, mi_yz

def gen_train_data_softrank(batchsize, seq_len, dim=1, com_num=20, reg=0.1):
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
        x = torch.from_numpy(joint_samples[:, 0]).unsqueeze(0)
        softrank_x = torchsort.soft_rank(x, regularization_strength=reg).squeeze(0).numpy()
        softrank_x = (softrank_x - np.min(softrank_x))/(np.max(softrank_x) - np.min(softrank_x))
        y = torch.from_numpy(joint_samples[:, 1]).unsqueeze(0)
        softrank_y = torchsort.soft_rank(y, regularization_strength=reg).squeeze(0).numpy()
        softrank_y = (softrank_y - np.min(softrank_y))/(np.max(softrank_y) - np.min(softrank_y))
        
        data[i, :, 0] = softrank_x
        data[i, :, 1] = softrank_y
    return data.astype(np.float32)

def gen_train_data_softrank_weighted(batchsize, seq_len, dim=1, com_num=10, reg=0.1):
    ### generated data has shape [batchsize, seq_len, dim*2], each one in the batch is a distribution
    data = np.zeros((batchsize, seq_len, dim*2))
    components_list = np.arange(1, 11)
    weights = np.arange(1, 11)
    for i in range(batchsize):
        num_components = np.random.choice(components_list, p=weights/weights.sum())
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
        x = torch.from_numpy(joint_samples[:, 0]).unsqueeze(0)
        softrank_x = torchsort.soft_rank(x, regularization_strength=reg).squeeze(0).numpy()
        softrank_x = (softrank_x - np.min(softrank_x))/(np.max(softrank_x) - np.min(softrank_x))
        y = torch.from_numpy(joint_samples[:, 1]).unsqueeze(0)
        softrank_y = torchsort.soft_rank(y, regularization_strength=reg).squeeze(0).numpy()
        softrank_y = (softrank_y - np.min(softrank_y))/(np.max(softrank_y) - np.min(softrank_y))
        
        data[i, :, 0] = softrank_x
        data[i, :, 1] = softrank_y
    return data.astype(np.float32)

def gen_train_data_softrank_weighted_pre(batchsize, seq_len, dim=1, com_num=10, reg=0.1):
    ### generated data has shape [batchsize, seq_len, dim*2], each one in the batch is a distribution
    data = np.zeros((batchsize, seq_len, dim*2))
    components_list = np.arange(1, 11)
    weights = np.arange(1, 11)
    for i in range(batchsize):
        num_components = np.random.choice(components_list, p=weights/weights.sum())
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

        for j in range(2):
            joint_samples[:, j] = (joint_samples[:, j] - np.min(joint_samples[:, j]))/(np.max(joint_samples[:, j]) - np.min(joint_samples[:, j]))
        x = torch.from_numpy(joint_samples[:, 0]).unsqueeze(0)
        softrank_x = torchsort.soft_rank(x, regularization_strength=reg).squeeze(0).numpy()
        softrank_x = (softrank_x - np.min(softrank_x))/(np.max(softrank_x) - np.min(softrank_x))
        y = torch.from_numpy(joint_samples[:, 1]).unsqueeze(0)
        softrank_y = torchsort.soft_rank(y, regularization_strength=reg).squeeze(0).numpy()
        softrank_y = (softrank_y - np.min(softrank_y))/(np.max(softrank_y) - np.min(softrank_y))
        
        data[i, :, 0] = softrank_x
        data[i, :, 1] = softrank_y
    return data.astype(np.float32)


