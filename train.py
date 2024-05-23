import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
from data import gen_train_data
from model.decoder import Decoder
from model.encoder import Encoder
from model.infonet import infonet
from model.query import Query_Gen_transformer
from scipy.stats import rankdata
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim import Adam
from util.epoch_timer import epoch_time

latent_dim = 256
latent_num = 256
input_dim = 2
batchsize = 32  #### num of distributions trained in one epoch
seq_len = 2000  #### number of data points sampled from each distribution
decoder_query_dim = 1000 #### size of lookup table
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.0001

encoder = Encoder(
    input_dim=input_dim,
    latent_num=latent_num,
    latent_dim=latent_dim,
    cross_attn_heads=8,
    self_attn_heads=16,
    num_self_attn_per_block=8,
    num_self_attn_blocks=1
)

decoder = Decoder(
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

model = infonet(encoder=encoder, decoder=decoder, query_gen = query_gen, decoder_query_dim = decoder_query_dim).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')

#model.apply(initialize_weights)

params = [{'params': model.parameters()}]
optimizer = optim.Adam(params, lr=learning_rate)

#model, optimizer = accelerator.prepare(model, optimizer)

def train(model, batch, optimizer, writer, ma_et, clip=1.0, ma_rate=0.01,iter_num=1, log_freq=10):
    model.train()
    epoch_loss = 0
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        losses.append(loss.item())
        
    return epoch_loss/iter_num, ma_et

def run(total_epoch, writer):
    train_losses= []
    ma_et = 1
    for step in range(total_epoch):
        start_time = time.time()
        #batch = gen_data_order(batchsize, seq_len, dim=1, corr=0.6, hidden_dim = 256)
        #batch = gen_vary_rank_noise(batchsize=batchsize, dim=1, com_num=20)
        batch = gen_train_data(batchsize=batchsize, seq_len=seq_len, dim=1, com_num=20)
        
        train_loss, ma_et = train(model, batch, optimizer, writer, ma_et)
        
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
    writer = SummaryWriter('logs/uniform')

    for i in range(0, 2000, 1):
        print(f"=========================================== epoch {i} begins!!! ===========================================")
        train_loss = run(total_epoch=100, writer=writer)
        
        if (i+1)%40==0:
            torch.save(model.module.state_dict(), f'saved/multi_marginal/model_{seq_len}_{batchsize}_{decoder_query_dim}-{i + 1}-{train_loss:.2f}.pt')