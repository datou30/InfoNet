# InfoNet PyTorch Implementation

This repository contains a PyTorch implementation of the article "InfoNet: An Efficient Feed-Forward Neural Estimator for Mutual Information." Please note that this implementation is an initial version and may be subject to further updates and improvements.

## Getting Started

### Training InfoNet from Scratch

To train an InfoNet model from scratch, you can use the `train.py` script. This script will guide you through the process of initializing and training your model using the default or custom datasets.

### Fine-Tuning on Pretrained Checkpoints

If you have a pretrained InfoNet model and wish to fine-tune it on a new dataset or improve its performance, the `train.py` script also supports fine-tuning. Simply provide the path to your existing checkpoint, and the script will resume training from there.

### Estimating Mutual Information

For applications involving mutual information estimation using InfoNet, the `infer.py` script is designed to facilitate this process. It allows you to estimate mutual information values based on your trained model and input data. Pre-trained checkpoint can be found in: [Download Checkpoint]([https://your-google-drive-link.com](https://drive.google.com/file/d/1AalM-qoUYsJ5SS38hznXHSIv5h8lKVDx/view?usp=sharing))

## Requirements

Before you begin, ensure you have all the necessary dependencies installed. A `requirements.txt` file is provided for convenience. You can install all required packages using the following command:

```bash
pip install -r requirements.txt
