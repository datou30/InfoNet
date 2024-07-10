# InfoNet: Neural Estimation of Mutual Information without Test-Time Optimization

Welcome to InfoNet 😀! This branch is a pytorch implementation of our paper [InfoNet: Neural Estimation of Mutual Information without Test-Time Optimization](https://arxiv.org/abs/2402.10158).
We replace the hard rank preprocessing with softrank and provide corresponding checkpoints here.

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

SoftRank provides a smooth approximation of traditional rank functions. It has a regularization parameter controls the smoothness of the ranking function: smaller values make it closer to the original rank, while larger values make it smoother. We utilize the method proposed in [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871), github repo can be found [here](https://github.com/teddykoker/torchsort).

```python 
pip install torchsort
x, y = np.random.multivariate_normal(mean=[0,0], cov=[[1,0.5],[0.5,1]], size=5000).T
x = torchsort.soft_rank(torch.from_numpy(x).unsqueeze(0), regularization_strength=1e-3)
```

<p align="center">
  <img src="https://drive.google.com/uc?id=1hXAl9qpjw9nO_H8SfuYvEcwkVPw8m-Gx" alt="softrank" width="280"/>
</p>



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
