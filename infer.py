import glob
import logging
import math
import pickle
import time

import numpy as np
import torch
import torch.autograd as autograd
from scipy.stats import rankdata
from sklearn.mixture import GaussianMixture
from torch import nn, optim
from torch.optim import Adam

from perceiver.decoder import PerceiverDecoder
from perceiver.encoder import PerceiverEncoder
from perceiver.perceiver import PerceiverIO
from perceiver.perceiver_lap import PerceiverIO_lap
from perceiver.query import Query_Gen
from perceiver.query_new import Query_Gen_transformer, Query_Gen_transformer_PE
from util.epoch_timer import epoch_time
from util.look_table import (lookup_value_2d, lookup_value_average,
                             lookup_value_bilinear, lookup_value_close,
                             lookup_value_grid)

latent_dim = 256
latent_num = 256
input_dim = 2
batchsize = 32  #### num of distributions trained in one epoch
seq_len = 5000  #### number of data points sampled from each distribution
decoder_query_dim = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = PerceiverEncoder(
    input_dim=input_dim,
    latent_num=latent_num,
    latent_dim=latent_dim,
    cross_attn_heads=8,
    self_attn_heads=16,
    num_self_attn_per_block=8,
    num_self_attn_blocks=1
)

decoder = PerceiverDecoder(
    q_dim=decoder_query_dim,
    latent_dim=latent_dim,
)

query_gen = Query_Gen_transformer(
    input_dim = input_dim,
    dim = decoder_query_dim
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = PerceiverIO(encoder=encoder, decoder=decoder, query_gen = query_gen, decoder_query_dim = decoder_query_dim).cuda(device=0)
model.load_state_dict(torch.load('saved/vary_seqlen/model_2000_48_1000-400--0.13.pt', map_location=device))
#model = nn.DataParallel(model, device_ids=[0, 1])
print(f'The model has {count_parameters(model):,} trainable parameters')

def infer(model, batch):
    model.eval()
    batch = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        mi_lb = model(batch)
    return mi_lb

def compute_smi_mean(sample_x, sample_y, model, proj_num, seq_len):
    dx = sample_x.shape[1]
    dy = sample_y.shape[1]
    results = []
    for i in range(proj_num//32):
        batch = np.zeros((32, seq_len, 2))
        ## 32 could be larger.
        for j in range(32):
            theta = np.random.randn(dx)
            phi = np.random.randn(dy)
            x_proj = np.dot(sample_x, theta)
            y_proj = np.dot(sample_y, phi)
            x_proj = rankdata(x_proj)/seq_len
            y_proj = rankdata(y_proj)/seq_len
            xy = np.column_stack((x_proj, y_proj))
            batch[j, :, :] = xy
        infer1 = infer(model, batch).cpu().numpy()
        mean_infer1 = np.mean(infer1)
        results.append(mean_infer1)

    return np.mean(np.array(results))

if __name__ == '__main__':
    
    d = 10
    mu = np.zeros(d)
    sigma = np.eye(d)
    sample_x = np.random.multivariate_normal(mu, sigma, 5000)
    sample_y = np.random.multivariate_normal(mu, sigma, 5000)

    ## If you do not want fixed sequence you could use checkpoint in saved/vary_seqlen
    ## sample_x and sample_y has shape [seq_len, d_x] and [seq_len, d_y]
    ## this code uses sliced mutual information, and here the proj_num influence the result a lot, larger the better
    result = compute_smi_mean(sample_x, sample_y, model, proj_num=1000, seq_len=5000)
    print(result)
    