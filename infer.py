import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from model.decoder import Decoder
from model.encoder import Encoder
from model.infonet import infonet
from model.query import Query_Gen_transformer
from scipy.stats import rankdata

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_model(config):
    encoder = Encoder(
        input_dim=config['model']['input_dim'],
        latent_num=config['model']['latent_num'],
        latent_dim=config['model']['latent_dim'],
        cross_attn_heads=config['model']['cross_attn_heads'],
        self_attn_heads=config['model']['self_attn_heads'],
        num_self_attn_per_block=config['model']['num_self_attn_per_block'],
        num_self_attn_blocks=config['model']['num_self_attn_blocks']
    )

    decoder = Decoder(
        q_dim=config['model']['decoder_query_dim'],
        latent_dim=config['model']['latent_dim'],
    )

    query_gen = Query_Gen_transformer(
        input_dim=config['model']['input_dim'],
        dim=config['model']['decoder_query_dim']
    )
    
    model = infonet(
        encoder=encoder,
        decoder=decoder,
        query_gen=query_gen,
        decoder_query_dim=config['model']['decoder_query_dim']
    ).to(device)
    
    return model

def load_model(config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)
    model = create_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def estimate_mi(model, x, y):
    ## x and y are 1 dimensional sequences
    model.eval()
    x = rankdata(x)/len(x)
    y = rankdata(y)/len(y)
    batch = torch.stack((torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), dim=1).unsqueeze(0).to(device) ## batch has shape [1, sequence length, 2]
    with torch.no_grad():
        mi_lb = model(batch)
    return mi_lb

def infer(model, batch):
    ### batch has shape [batchsize, seq_len, 2]
    model.eval()
    batch = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():

        mi_lb = model(batch)
        MI = torch.mean(mi_lb)

    return MI.cpu().numpy()

def compute_smi_mean(sample_x, sample_y, model, proj_num, seq_len, batchsize):
    ## we use sliced mutual information to estimate high dimensional correlation
    ## proj_num means the number of random projections you want to use, the larger the more accuracy but higher time cost
    ## seq_len means the number of samples used for the estimation
    ## batchsize means the number of one-dimensional pairs estimate at one time, this only influences the estimation speed
    dx = sample_x.shape[1]
    dy = sample_y.shape[1]
    results = []
    for i in range(proj_num//batchsize):
        batch = np.zeros((batchsize, seq_len, 2))
        for j in range(batchsize):
            theta = np.random.randn(dx)
            phi = np.random.randn(dy)
            x_proj = np.dot(sample_x, theta)
            y_proj = np.dot(sample_y, phi)
            x_proj = rankdata(x_proj)/seq_len
            y_proj = rankdata(y_proj)/seq_len
            xy = np.column_stack((x_proj, y_proj))
            batch[j, :, :] = xy
        infer1 = infer(model, batch)
        mean_infer1 = np.mean(infer1)
        results.append(mean_infer1)

    return np.mean(np.array(results))

def example_d_1():
    seq_len = 4781
    results = []
    real_MIs = []
    
    for rou in np.arange(-0.9, 1, 0.1):
        x, y = np.random.multivariate_normal(mean=[0,0], cov=[[1,rou],[rou,1]], size=seq_len).T
        x = rankdata(x)/seq_len #### important, data preprocessing is needed, using rankdata(x)/seq_len to map x and y to [0,1]
        y = rankdata(y)/seq_len
        result = estimate_mi(model, x, y).squeeze().cpu().numpy()
        real_MI = -np.log(1-rou**2)/2
        real_MIs.append(real_MI)
        results.append(result)
        print("estimate mutual information is: ", result, "real MI is ", real_MI  )

def example_highd():
    d = 10
    mu = np.zeros(d)
    sigma = np.eye(d)
    sample_x = np.random.multivariate_normal(mu, sigma, 2000)
    sample_y = np.random.multivariate_normal(mu, sigma, 2000)
    result = compute_smi_mean(sample_x, sample_y, model, seq_len=2000, proj_num=1024, batchsize=32)
    print(f"result is {result}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Load model from config and checkpoint')
    parser.add_argument('--config', type=str, required=False, default='configs/config.yaml', help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, required=False, default="saved/uniform/model_5000_32_1000-720--0.16.pt", help='Path to the model checkpoint')
    
    args = parser.parse_args()
    
    model = load_model(args.config, args.checkpoint)
    print("Model loaded successfully")

    example_d_1()
    #example_highd()
