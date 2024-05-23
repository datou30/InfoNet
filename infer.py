import matplotlib.pyplot as plt
import numpy as np
import torch
from model.decoder import Decoder
from model.encoder import Encoder
from model.infonet import infonet
from model.query import Query_Gen_transformer
from scipy.stats import rankdata

latent_dim = 256
latent_num = 256
input_dim = 2
decoder_query_dim = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model = infonet(encoder=encoder, decoder=decoder, query_gen = query_gen, decoder_query_dim = decoder_query_dim).to(device)
model.load_state_dict(torch.load('saved/uniform/model_5000_32_1000-720--0.16.pt', map_location=device))

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
        xy = np.column_stack((x, y))
        result = infer(model, xy.reshape(1, seq_len, 2))
        real_MI = -np.log(1-rou**2)/2
        real_MIs.append(real_MI)
        results.append(result)
        print("estimate mutual information is: ", result, "real MI is ", real_MI  )

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(10,10))

    ax1 = fig.add_subplot(111)
        
    ax1.plot(np.arange(-0.9, 1, 0.1), np.array(real_MIs), color="red", lw=2, ls="-", label="real mutual information",  markersize=10)
    ax1.plot(np.arange(-0.9, 1, 0.1), np.array(results), color="blue", lw=2, ls="-", label="estimate MI",   markersize=10)
    ax1.set_xlabel("# rou", fontweight="bold", fontsize=20)
    ax1.set_ylabel(" mutual information ", fontweight="bold", fontsize=20)
    ax1.legend(fontsize=20, loc="upper left")

    plt.savefig(f"./plots/gauss_result.png")

def example_highd():
    d = 10
    mu = np.zeros(d)
    sigma = np.eye(d)
    sample_x = np.random.multivariate_normal(mu, sigma, 2000)
    sample_y = np.random.multivariate_normal(mu, sigma, 2000)
    result = compute_smi_mean(sample_x, sample_y, model, seq_len=2000, proj_num=1024, batchsize=32)
    print(result)

if __name__ == '__main__':

    ### infonet directly predict MI between 1-dimensional variables
    ### we apply sliced mutual information for high dimension
    example_d_1()
    #example_highd()
    