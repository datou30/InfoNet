# InfoNet PyTorch Implementation

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1CS-iVGgRriB3Erl4fn8fLUJOjf6BHNqd" alt="infonet_logo" width="280"/>
</p>

Welcome to InfoNet 😀! This is a PyTorch implementation of the article [InfoNet: Neural Estimation of Mutual Information without Test-Time Optimization](https://arxiv.org/abs/2402.10158).
Our project page can be found [here](https://datou30.github.io/InfoNet-page/).
You can utilize it to compute mutual information between two sequences quickly!

## Method overview

Mutual information (MI) is a valuable metric for assessing the similarity between two variables and has lots of applications in deep learning. However, current neural MI estimation methods such as [MINE](https://arxiv.org/abs/1801.04062) are a bit time-costly (needs over 1 minute to get the estimation between pairs of data). Our work is to extremely make the process much faster. 

To overcome this, our method trains a pre-trained model InfoNet using various synthetic distributions. Unlike MINE requires training an MLP from scratch separately for every pair of data, our model only needs one feed-forward process to get the estimation of MI. Experiments have shown our method has strong generalization ability on unseen distributions.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1aJ-WDgvQRoCTBp7HLsMEOCIKhZG_lm-1" alt="infonet_logo" width="300"/>
</p>

## Getting Started

### Requirements

This is a requirement list, ensure you have all the necessary dependencies installed. You can install all required packages using:

```bash
pip install -r requirements.txt
```

### Training InfoNet from Scratch

To train the model from scratch or finetune on specific distributions, `train.py` provides an example. This script will guide you through the process of initializing and training your model using the default Gaussian mixture distribution dataset. It will take about 4 hours to get convergence on 2 RTX 4090 GPU.

### Estimating Mutual Information

When doing inference, estimating mutual information estimation using InfoNet, you can follow examples from `infer.py`. 
Note that inputs should have shape [batchsize, sequence length, 2]. InfoNet is capable of estimating MI between multiple pairs at one time. 
However, current version of InfoNet could only estimate direct MI between two 1-dimensional variables.
For high dimensional correlation estimation, we apply [Sliced Mutual Information](https://arxiv.org/abs/2110.05279) instead, this retains the structural properties of classic MI while offering scalable computation, efficient high-dimensional estimation, and enhanced feature extraction capabilities.

Pre-trained checkpoint can be found in: [Download Checkpoint](https://drive.google.com/drive/folders/1R7ah_ymD3M9Fp9EegyJrWNo5hI6Z5gZ7?usp=drive_link)

### Data Preprocessing

Data preprocessing is crucial in the estimation result of InfoNet. In order to get the correct result, you need to apply copula transformation(rankdata) on each dimension before pushing the data into the network.
Also, rankdata will lead to undifferentiable, if you want to apply InfoNet in the training task, you can utilize differentiable rank techniques such as [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871).

For high-dimensional estimation using sliced mutual information, we have found first applying a linear mapping on each dimension separately (e.g. map all the dimensions between -1 and 1) before doing random projections will increase the performance.

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
