# InfoNet: Neural Estimation of Mutual Information without Test-Time Optimization

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1CS-iVGgRriB3Erl4fn8fLUJOjf6BHNqd" alt="infonet_logo" width="280"/>
</p>

Welcome to InfoNet 😀! This is a PyTorch implementation of our paper [InfoNet: Neural Estimation of Mutual Information without Test-Time Optimization](https://arxiv.org/abs/2402.10158).
Our project page can be found [here](https://datou30.github.io/InfoNet-page/).
You can utilize it to compute mutual information between two sequences quickly!

(This project is currently under active development. We are continuously working on perfecting this repo and the project page.✨)

## Method overview

Mutual information (MI) is a valuable metric for assessing the similarity between two variables and has lots of applications in deep learning. However, current neural MI estimation methods such as [MINE](https://arxiv.org/abs/1801.04062) are a bit time-costly (needs over 1 minute to get the estimation between pairs of data). Our work is to utilize a neural method to make the process much faster. 

In order to achieve this, our method designs a network structure and trains a pre-trained model InfoNet using various synthetic distributions, it can remember various information from a large amount of distributions. Unlike MINE, which requires training an MLP from scratch separately for every pair of data when estimating MI, our model only needs to do one forward pass to get the estimation. Experiments have shown our method has strong generalization ability on unseen distributions.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1aJ-WDgvQRoCTBp7HLsMEOCIKhZG_lm-1" alt="infonet_logo" width="300"/>
</p>

## Getting Started

### Requirements

This is a requirement list, ensure you have all the necessary dependencies installed. You can install all required packages using:

```bash
pip install -r requirements.txt
```

### Estimating Mutual Information

When doing inference, estimating mutual information estimation using InfoNet, you can follow examples below or refer to `infer.py` for additional instruction.

when x and y are scalable random variables:
```python 
import numpy as np
import torch
from scipy.stats import rankdata
from infer import load_model, estimate_mi, compute_smi_mean

config_path = "configs/config.yaml"
ckpt_path = "saved/uniform/model_5000_32_1000-720--0.16.pt"
model = load_model(config_path, ckpt_path)

## random generate gauss distribution examples
seq_len = 4781
rou = 0.5
x, y = np.random.multivariate_normal(mean=[0,0], cov=[[1,rou],[rou,1]], size=seq_len).T

## data preprocessing and estimating
x = rankdata(x)/seq_len
y = rankdata(y)/seq_len
result = estimate_mi(model, x, y).squeeze().cpu().numpy()
real_MI = -np.log(1-rou**2)/2
print("estimate mutual information is: ", result, "real MI is ", real_MI)
```

If x and y are high-dimensional variables, we apply [Sliced Mutual Information](https://arxiv.org/abs/2110.05279) instead:

```python 
d = 10
mu = np.zeros(d)
sigma = np.eye(d)
sample_x = np.random.multivariate_normal(mu, sigma, 2000)
sample_y = np.random.multivariate_normal(mu, sigma, 2000)
result = compute_smi_mean(sample_x, sample_y, model, seq_len=2000, proj_num=1024, batchsize=32)
## proj_num means the number of random projections you want to use, the larger the more accuracy but higher time cost
## seq_len means the number of samples used for the estimation
## batchsize means the number of one-dimensional pairs estimate at one time, this only influences the estimation speed
```

Note that inputs should have shape [batchsize, sequence length, 2]. InfoNet is capable of estimating MI between multiple pairs at one time. 
Pre-trained checkpoint can be found in: [Download Checkpoint](https://drive.google.com/drive/folders/1R7ah_ymD3M9Fp9EegyJrWNo5hI6Z5gZ7?usp=drive_link)

### Training InfoNet from Scratch

To train the model from scratch or finetune on specific distributions, `train.py` provides an example. This script will guide you through the process of initializing and training your model using the default Gaussian mixture distribution dataset. It will take about 4 hours to get convergence on 2 RTX 4090 GPU.

### Data Preprocessing

Data preprocessing is crucial in the estimation result of InfoNet. You should make sure to use the same data preprocessing method in the training and testing (e.g. using copula transformation or linear scaling).
```python 
from scipy.stats import rankdata
x = rankdata(x)/seq_len
y = rankdata(y)/seq_len
```
Also, `rankdata` will lead to undifferentiable, if you want to apply InfoNet in the training task, you can replace `rankdata` with differentiable rank techniques： [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871), [github repo link here](https://github.com/teddykoker/torchsort). 
Detailed instruction of softrank will be shown in the new branch softrank.
```python 
pip install torchsort

x, y = np.random.multivariate_normal(mean=[0,0], cov=[[1,0.5],[0.5,1]], size=5000).T
x = torchsort.soft_rank(torch.from_numpy(x).unsqueeze(0), regularization_strength=1e-3)/5000
y = torchsort.soft_rank(torch.from_numpy(y).unsqueeze(0), regularization_strength=1e-3)/5000
```

For high-dimensional estimation using sliced mutual information, we have found first applying a linear mapping on each dimension separately (e.g. map all the dimensions between -1 and 1) before doing random projections will increase the performance.

```python 
## linear scale [batchsize, seq_len, dim] to [-1,1] on seq_len
min_val = torch.min(input_tensor, dim=1, keepdim=True).values
max_val = torch.max(input_tensor, dim=1, keepdim=True).values
scaled_tensor = 2 * (input_tensor - min_val) / (max_val - min_val) - 1
```
### Evaluation Dataset

In `gmm_eval_dataset`, we have provided a series of parameters for Gaussian Mixture Models along with the ground truth mutual information between X and Y. They are categorized according to the number of Gaussian components, each with 5000 random generated distributions.

### Experiments 

Experiments can be found in `Notebooks`, we provide two .ipynb files to reproduce our experimental results detailedly.

### Acknowledgement

We would like to express our gratitude to [esceptico/perceiver-io](https://github.com/esceptico/perceiver-io) for providing the code base that significantly assisted in the development of our program.

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
