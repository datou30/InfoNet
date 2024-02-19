import glob
import logging
import math
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import torch
import torch.autograd as autograd
from data import *
from data_gen import *
from perceiver.decoder import PerceiverDecoder
from perceiver.encoder import PerceiverEncoder
from perceiver.perceiver import PerceiverIO
from perceiver.perceiver_lap import PerceiverIO_lap
from perceiver.query import Query_Gen
from perceiver.query_new import Query_Gen_transformer, Query_Gen_transformer_PE
from sklearn.mixture import GaussianMixture
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim import Adam
from util.epoch_timer import epoch_time
from util.look_table import (lookup_value_2d, lookup_value_average,
                             lookup_value_bilinear, lookup_value_close,
                             lookup_value_grid)

latent_dim = 256
latent_num = 256
input_dim = 2
batchsize = 64  #### num of distributions trained in one epoch
seq_len = 2000  #### number of data points sampled from each distribution
decoder_query_dim = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.0004

encoder = PerceiverEncoder(
    input_dim=input_dim,
    latent_num=latent_num,
    latent_dim=latent_dim,
    cross_attn_heads=8,
    self_attn_heads=8,
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

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]

model = PerceiverIO(encoder=encoder, decoder=decoder, query_gen = query_gen, decoder_query_dim = decoder_query_dim).cuda(device=0)
model = nn.DataParallel(model, device_ids=[0, 1])
#model.load_state_dict(torch.load('saved/marginal/model_2000_32_1000-200--0.14.pt', map_location=device))
print(f'The model has {count_parameters(model):,} trainable parameters')

#model.apply(initialize_weights)


params = [{'params': model.parameters()}]
optimizer = optim.Adam(params, lr=learning_rate)

def train(model, batch, optimizer, writer, clip=1.0, ma_rate=0.01,iter_num=1, log_freq=10):
    model.train()
    epoch_loss = 0
    ma_et = 1
    losses = []
    global global_step
    for i in range(iter_num):
        
        global_step += 1

        batch = torch.tensor(batch, dtype=torch.float32, device=device)

        mi_lb = model(batch)
    
        loss = - torch.mean(mi_lb) 
        
        if (global_step)%(log_freq)==0:            
            writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=global_step)
            
        optimizer.zero_grad()
        autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        losses.append(loss.item())
        
    return epoch_loss/iter_num

def run(total_epoch, batch, writer):
    train_losses= []
    for step in range(total_epoch):
        start_time = time.time()

        train_loss = train(model, batch, optimizer, writer)
        
        end_time = time.time()

        train_losses.append(train_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        #if step==(total_epoch-1):
            #torch.save(model.state_dict(), 'saved/model-normal-{0}.pt'.format(train_loss))

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} ')

    return train_loss

def infer(model, batch):
    model.eval()
    batch = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():

        mi_lb = model(batch)
        MI = torch.mean(mi_lb)

    return MI
    
if __name__ == '__main__':
    ma_rate = 1.0
    global_step = 0
    writer = SummaryWriter('logs/experiment_test')
    
    for i in range(0, 300000, 1):
        
        ###################################################################
        # fine tune step, train data has shape [batchsize, sequence_length, 2]
        # suppose you have high dimensional data, then the train data uses random linear projection into one dimension
        batch = batch #### prepare your batch here

        train_loss = run(total_epoch=1, batch=batch, writer=writer)
        
        