import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from model.decoder import Decoder
from model.encoder import Encoder
from model.infonet import infonet
from model.query import Query_Gen_transformer
from scipy.stats import rankdata
import lightning
import hydra
import torchsort
from einops import rearrange

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class infonet_lightning(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        encoder = Encoder(
            input_dim=input_dim,
            latent_num=latent_num,
            latent_dim=latent_dim,
            cross_attn_heads=8,
            self_attn_heads=16,
            num_self_attn_per_block=8,
            num_self_attn_blocks=1,
        )

        decoder = Decoder(
            q_dim=decoder_query_dim,
            latent_dim=latent_dim,
        )

        query_gen = Query_Gen_transformer(
            input_dim=input_dim,
            dim=decoder_query_dim,
        )
        self.model = infonet(encoder=encoder, decoder=decoder, query_gen=query_gen, decoder_query_dim=decoder_query_dim)

    def forward(self, x):
        return self.model(x)

def load_model(config_path: str, config_name: str, ckpt_path: str) -> infonet_lightning:
    
    hydra.initialize(config_path=config_path)
    cfg = hydra.compose(config_name=config_name)

    global latent_dim, latent_num, input_dim, batchsize, seq_len, decoder_query_dim, max_input_dim
    batchsize = cfg.batchsize
    latent_dim = cfg.latent_dim
    latent_num = cfg.latent_num
    input_dim = cfg.input_dim
    seq_len = cfg.seq_len
    decoder_query_dim = cfg.decoder_query_dim

    torch.set_float32_matmul_precision('medium')

    model = infonet_lightning.load_from_checkpoint(
        ckpt_path,
        map_location=lambda storage, loc: storage.cuda(0)
    )

    return model

def softrank_preprocessing(input_tensor, regularization_strength=0.1):
    ## x has shape [batchsize, seq_len, 2]
    min_val = torch.min(input_tensor, dim=1, keepdim=True).values
    max_val = torch.max(input_tensor, dim=1, keepdim=True).values
    scaled_tensor = (input_tensor - min_val) / (max_val - min_val)
    
    softrank = torchsort.soft_rank(rearrange(scaled_tensor, 'b n d -> (d b) n'), regularization_strength=regularization_strength)
    softrank = rearrange(softrank, '(d b) n -> b n d', d=2)

    min_val = torch.min(softrank, dim=1, keepdim=True).values
    max_val = torch.max(softrank, dim=1, keepdim=True).values
    scaled_softrank = (softrank - min_val) / (max_val - min_val) 

    return scaled_softrank

def estimate_mi(module, x, y):
    ## x and y are 1 dimensional sequences
    module.eval()
    batch = torch.stack((torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), dim=1).unsqueeze(0).to(device) ## batch has shape [1, sequence length, 2]
    batch = softrank_preprocessing(batch)
    with torch.no_grad():
        mi_lb = module(batch)
    return mi_lb

def compute_smi_mean(sample_x, sample_y, module, proj_num, batchsize):
    ## we use sliced mutual information to estimate high dimensional correlation
    ## proj_num means the number of random projections you want to use, the larger the more accuracy but higher time cost
    ## seq_len means the number of samples used for the estimation
    ## batchsize means the number of one-dimensional pairs estimate at one time, this only influences the estimation speed
    module.eval()
    dx = sample_x.shape[1]
    dy = sample_y.shape[1]
    seq_len = sample_x.shape[0]
    results = []
    for i in range(proj_num//batchsize):
        
        batch = torch.zeros((batchsize, seq_len, 2)).to(device)
        theta = torch.randn(batchsize, dx).to(device)  
        phi = torch.randn(batchsize, dy).to(device)   

        x_proj = torch.matmul(sample_x, theta.T)  # [seq_len, batchsize]
        y_proj = torch.matmul(sample_y, phi.T)    # [seq_len, batchsize]

        x_proj = x_proj.permute(1, 0)  # [batchsize, seq_len]
        y_proj = y_proj.permute(1, 0)  # [batchsize, seq_len]

        batch = torch.stack((x_proj, y_proj), dim=2)  # [batchsize, seq_len, 2]
        batch = softrank_preprocessing(batch)
        with torch.no_grad():
            infer1 = module(batch)
        mean_infer1 = torch.mean(infer1)
        results.append(mean_infer1)

    return torch.mean(torch.tensor(results))

def example_d_1(module):

    module.eval()
    seq_len = 4781
    results = []
    real_MIs = []
    
    for rou in np.arange(-0.9, 1, 0.1):
        sample = np.random.multivariate_normal(mean=[0,0], cov=[[1,rou],[rou,1]], size=seq_len).T.astype(np.float32)
        sample = torch.from_numpy(sample).unsqueeze(0)
        sample = rearrange(sample, 'b d n -> b n d')
        sample = softrank_preprocessing(sample).to(device)
        with torch.no_grad():
            result = module(sample).squeeze().cpu().numpy()
        real_MI = -np.log(1-rou**2)/2
        real_MIs.append(real_MI)
        results.append(result)
        print("estimate mutual information is: ", result, "real MI is ", real_MI  )

def example_highd(module):
    d = 10
    mu = np.zeros(d)
    sigma = np.eye(d)
    sample_x = torch.from_numpy(np.random.multivariate_normal(mu, sigma, 2000).astype(np.float32)).to(device)
    sample_y = torch.from_numpy(np.random.multivariate_normal(mu, sigma, 2000).astype(np.float32)).to(device)
    result = compute_smi_mean(sample_x, sample_y, module, proj_num=1024, batchsize=64)
    print(f"result is {result}")

if __name__ == "__main__":
    config_path = 'configs'
    config_name = 'cfg_softrank_0.1_new'
    ckpt_path = '/home/hzy/work/kNN/infonet_softrank/saved/basic_hypernet-epoch=1399-val_acc=98.93.ckpt'
    model = load_model(config_path, config_name, ckpt_path)
    print("Model loaded successfully")
    example_d_1(model)
    #example_highd(model)
    #x, y = np.random.multivariate_normal(mean=[0,0], cov=[[1,0.6],[0.6,1]], size=seq_len).T.astype(np.float32)
    #mi = estimate_mi(model, x, y)