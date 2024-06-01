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

To train an InfoNet model from scratch, you can use the `train.py` script. This script will guide you through the process of initializing and training your model using the default Gaussian mixture dataset.

### Estimating Mutual Information

For applications involving mutual information estimation using InfoNet, the `infer.py` script provides several examples. Pre-trained checkpoint can be found in: [Download Checkpoint](https://drive.google.com/file/d/1AalM-qoUYsJ5SS38hznXHSIv5h8lKVDx/view?usp=sharing)

### Data Preprocessing

Data preprocessing is crucial in the estimation result of InfoNet. In order to get the correct result, you need to apply copula transformation(rankdata) before pushing the data into the network.
Also, rankdata will lead to undifferentiable, if you want to apply InfoNet in the training task, you will need to use [differentiable rank](https://arxiv.org/abs/2002.08871).

For high-dimensional estimation using sliced mutual information, we have find first applying a linear mapping on each dimension separately (e.g. map all the dimensions between -1 and 1) before doing random slices will increase the performance.

## Method Discription

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
