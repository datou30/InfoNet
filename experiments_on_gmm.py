import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob, _compute_precision_cholesky, _estimate_gaussian_covariances_full
from scipy.special import logsumexp
import torch
from model.decoder import Decoder
from model.encoder import Encoder
from model.infonet import infonet
from model.query import Query_Gen_transformer
from scipy.stats import rankdata

import os

def score_samples_marginal(X, gm, index):
    ## Compute the log-likelihood of each sample for the marginal model,
    ## in 1-D the Cholesky decomposition is simply the inverse sqrt of the variance
    oned_cholesky = np.sqrt(1 / gm.covariances_[:, index, index]).reshape(-1, 1, 1)
    marginal_logprob = _estimate_log_gaussian_prob(
        X, gm.means_[:, index].reshape(-1, 1), oned_cholesky, gm.covariance_type
    )
    return logsumexp(np.log(gm.weights_) + marginal_logprob, axis=1)

def gen_gmm_withmi(num_components, num_samples=1e7):
    ## generate mixture of gaussian distributions with estimated mi
    ## num_samples is used to estimate MI, the larger the more accurate estimation
    num_components = num_components
    weights = np.random.dirichlet(np.ones(num_components))

    means = [np.random.uniform(-5, 5, size=2) for _ in range(num_components)]
    covs = []
    for _ in range(num_components):
        A = np.random.uniform(-3, 3, size=(2, 2))
        cov = np.dot(A, A.transpose()) + 0.01 * np.eye(2)
        covs.append(cov)

    gm = GaussianMixture(n_components=num_components)
    gm.weights_ = np.array(weights)
    gm.means_ = np.array(means)
    gm.covariances_ = np.array(covs)

    samples, labels = gm.sample(n_samples=num_samples)
    samples = np.array(samples)

    gm.precisions_cholesky_ = _compute_precision_cholesky(
        gm.covariances_, gm.covariance_type
    )

    joint_xy = gm.score_samples(samples)
    marginal_x = score_samples_marginal(samples[:, [0]], gm, index=0)
    marginal_y = score_samples_marginal(samples[:, [1]], gm, index=1)

    MI_xy = np.mean(joint_xy - marginal_x - marginal_y)
    return gm, MI_xy

def infer(model, batch):
    ### batch has shape [batchsize, seq_len, 2]
    model.eval()
    batch = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():

        mi_lb = model(batch)
        MI = torch.mean(mi_lb)

    return MI.cpu().numpy()

def main():
    latent_dim = 256
    latent_num = 256
    input_dim = 2
    decoder_query_dim = 1000
    global device
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

    num_components = 5
    for _ in range(10):
        gm, mi = gen_gmm_withmi(num_components)
        test_samples, labels = gm.sample(n_samples=5000)
        test_samples[:, 0] = rankdata(test_samples[:, 0])/test_samples.shape[0]
        test_samples[:, 1] = rankdata(test_samples[:, 1])/test_samples.shape[0]
        test_samples = np.expand_dims(test_samples, axis=0)
        est_mi = infer(model, test_samples)
        print("estimate mutual information is: ", est_mi, "real MI is ", mi  )

if __name__ == '__main__':
    main()