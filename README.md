# InfoNet: Neural Estimation of Mutual Information without Test-Time Optimization

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1CS-iVGgRriB3Erl4fn8fLUJOjf6BHNqd" alt="infonet_logo" width="280"/>
</p>

Welcome to InfoNet 😀! This is a PyTorch implementation of our paper [InfoNet: Neural Estimation of Mutual Information without Test-Time Optimization](https://arxiv.org/abs/2402.10158).
Our project page can be found [here](https://datou30.github.io/InfoNet-page/).
You can utilize it to compute mutual information between two sequences quickly!

(This project is currently under active development. We are continuously working on perfecting this repo and the project page.✨)

## Getting Started

### Estimating Mutual Information

When doing inference, estimating mutual information estimation using InfoNet, you can follow examples below or refer to `infer.py` for additional instruction.

load checkpoints:
```python 
import torch
import numpy as np
from infer import load_model

config_path = 'configs'
config_name = 'cfg_softrank_0.1_new'
ckpt_path = 'path/to/ckpt'
model = load_model(config_path, config_name, ckpt_path)
```

when x and y are scalable random variables:
```python
from infer import estimate_mi
## random generate gauss distribution examples
seq_len = 4781
rou = 0.5
x, y = np.random.multivariate_normal(mean=[0,0], cov=[[1,rou],[rou,1]], size=seq_len).T

result = estimate_mi(model, x, y).squeeze().cpu().numpy()
real_MI = -np.log(1-rou**2)/2
print("estimate mutual information is: ", result, "real MI is ", real_MI)
```

If x and y are high-dimensional variables, using [Sliced Mutual Information](https://arxiv.org/abs/2110.05279) instead:

```python
from infer import compute_smi_mean 
d = 10
mu = np.zeros(d)
sigma = np.eye(d)
sample_x = np.random.multivariate_normal(mu, sigma, 2000)
sample_y = np.random.multivariate_normal(mu, sigma, 2000)
result = compute_smi_mean(sample_x, sample_y, model, proj_num=1024, batchsize=32)
## proj_num means the number of random projections you want to use, the larger the more accuracy but higher time cost
## batchsize means the number of one-dimensional pairs estimate at one time, this only influences the estimation speed
```

Note that inputs should have shape [batchsize, sequence length, 2]. InfoNet is capable of estimating MI between multiple pairs at one time. 
Pre-trained checkpoint can be found in: [Download Softrank Checkpoint](https://drive.google.com/drive/folders/1lP8EvS-Gg146rDZrW1TTmXEuHqA6CW85?usp=drive_link).

### Training InfoNet from Scratch

To train the model from scratch or finetune on specific distributions, `train_softrank.py` provides an example. This script will guide you through the process of initializing and training your model using the default Gaussian mixture distribution dataset. It will take about 8 hours to get convergence on 2 RTX 4090 GPU.

### Brief Introduction of Soft Rank

For high-dimensional estimation using sliced mutual information, we have found first applying a linear mapping on each dimension separately (e.g. map all the dimensions between -1 and 1) before doing random projections will increase the performance.

```python 
## linear scale [batchsize, seq_len, dim] to [-1,1] on seq_len
min_val = torch.min(input_tensor, dim=1, keepdim=True).values
max_val = torch.max(input_tensor, dim=1, keepdim=True).values
scaled_tensor = 2 * (input_tensor - min_val) / (max_val - min_val) - 1
```

## Citing Our Work

If you find our work interesting and useful, please consider citing our paper:
```bash
@article{hu2024infonet,
  title={InfoNet: Neural Estimation of Mutual Information without Test-Time Optimization},
  author={Hu, Zhengyang and Kang, Song and Zeng, Qunsong and Huang, Kaibin and Yang, Yanchao},
  journal={arXiv preprint arXiv:2402.10158},
  year={2024}
}
```
