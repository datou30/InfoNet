import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import time

import matplotlib.pyplot as plt

class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(Mine(hidden_size=256)))

def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    # print(t.shape, marginal.shape)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint, marginal = batch
    joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
    marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

    # unbiasing use moving average

    loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
    # loss = - mi_lb
    # print(torch.mean(t), torch.mean(et), (1/ma_et.mean()).detach()*torch.mean(et))
    # print(loss.item())
    mine_net_optim.zero_grad()
    autograd.backward(loss)
    mine_net_optim.step()
    return mi_lb, ma_et, loss.item()


def sample_batch(data, batch_size, sample_mode='joint'):
    dim = data.shape[1]//2
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]
    else:
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = np.concatenate([data[joint_index][:, 0:dim],
                                data[marginal_index][:, dim:2*dim]],
                               axis=1)
    return batch


def train(data, mine_net, mine_net_optim, batch_size=100, iter_num=int(100)):
    # data is x or y
    result = list()
    ma_et = 1.
    for i in range(iter_num):
        batch = sample_batch(data, batch_size=batch_size) \
            , sample_batch(data, batch_size=batch_size, sample_mode='marginal')
        mi_lb, ma_et, train_loss = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        result.append(mi_lb.detach().cpu().numpy())
    return result


def ma(a, window_size=50):
    return [np.mean(a[i:i + window_size]) for i in range(0, len(a) - window_size)]


def scale_data(input_tensor):
    min_val = np.min(input_tensor)
    max_val = np.max(input_tensor)
    scaled_tensor = 2 * (input_tensor - min_val) / (max_val - min_val) - 1
    return scaled_tensor

def MINE_esti(data, batch_size=100, iter_num=int(100)):

    input_size = data.shape[1]
    mine_net = Mine(input_size=input_size).cuda()
    mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-2)
    result = train(data, mine_net, mine_net_optim, batch_size=batch_size, iter_num=iter_num)
    result_ma = ma(result, window_size=50)
    final_result = result_ma[-1]
    return final_result

if __name__ == '__main__':

    times = []
    results = []
    real_MIs = []
    for rou in np.arange(-0.9, 1, 0.1):
        data = np.random.multivariate_normal(mean=[0,0], cov=[[1,rou],[rou,1]], size=2000)
        print(data.shape)
        start_time = time.time()
        result = MINE_esti(data, batch_size=100, iter_num=int(1000))
        end_time = time.time()
        real_MI = -np.log(1-rou**2)/2
        real_MIs.append(real_MI)
        times.append(end_time-start_time)
        results.append(result)
        print("data", rou, "over, time is", end_time - start_time)
        print("estimate mutual information is: ", result, "real MI is ", real_MI  )
        