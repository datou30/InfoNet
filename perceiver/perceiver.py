from typing import Optional

import torch
import torch.nn as nn

from .encoder import PerceiverEncoder
from .decoder import PerceiverDecoder
from .query_new import Query_Gen_transformer, Query_Gen_transformer_PE
from .query import Query_Gen
from .gauss_mild import GaussConv
import torch.nn.functional as F
import numpy as np
#from .query import Query_Gen

def lookup_value_grid(matrix, vector, mode):
    
    #matrix = matrix.squeeze(0)
    #vector = vector.squeeze(0)
    batchsize = matrix.shape[0]
    dim = matrix.shape[1]
    num = vector.shape[1]
    matrix = matrix.view(batchsize, 1, dim, dim)
    vector = vector.view(batchsize, num, 1, 2)
    value = (F.grid_sample(matrix, vector, mode, padding_mode='border', align_corners=True)).squeeze()
    return value   

def mutual_information(batch, matrix):
    ## batch is only joint
    perm = torch.randperm(batch.shape[1])
    marginal = torch.cat((batch[:, :, 0].unsqueeze(2), batch[:, perm, 1].unsqueeze(2)), dim=2)
    #reshape_merge = torch.reshape(merge, (1,200,100))
    #j_matrix = model(joint)  
    #m_matrix = model(marginal)
    #print('===================> matrix range is:', torch.max(j_matrix), torch.min(j_matrix))
    
    t = lookup_value_grid(matrix, batch, "bilinear")
    t = torch.clamp(t, min=-200, max=200)
    #print('===================> t range is:', torch.max(t), torch.min(t) ,'\n')
    et = torch.exp(torch.clamp(lookup_value_grid(matrix, marginal, "bilinear"), min=-200, max=200))
    
    #print('===================> et range is:', torch.max(et), torch.min(et),'\n')
    #t_m = lookup_value_grid(matrix, marginal, "bilinear")
    if len(t.shape)==1:
        t = t.unsqueeze(0)
        et = et.unsqueeze(0)
    mi_lb = torch.mean(t, dim=1) - torch.log(torch.mean(et, dim=1))
    if torch.isnan(- torch.mean(mi_lb) ):
        raise ValueError("Loss is NaN!")
    return mi_lb

class PerceiverIO(nn.Module):

    def __init__(
        self,
        encoder: PerceiverEncoder,
        decoder: PerceiverDecoder,
        query_gen: Query_Gen_transformer,
        decoder_query_dim: int
    ):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.query_gen = query_gen
        self.mild = GaussConv(size=15, nsig=3, channels=1)
        
        #self.query = nn.Parameter(torch.randn(1, decoder_query_dim, decoder_query_dim))

    def forward(
        self,
        inputs: Optional[torch.Tensor],
        query: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None
    ):
    
        latents = self.encoder(inputs, input_mask)
        query = self.query_gen(inputs)
        
        outputs = self.decoder(
            x_q=query,
            latents=latents,
            query_mask=query_mask
        )
        #print(outputs.shape)
        #torch.save(outputs.cpu().numpy(), "lookuptable.pth")
        outputs = outputs.unsqueeze(1)
        outputs = self.mild(outputs)
        outputs = outputs.squeeze(1)
        #torch.save(outputs.cpu().numpy(), "lookuptable_smoothed.pth")
        #print("saved!!!!!!!!!!!!!!!1")
        mi_lb = mutual_information(inputs, outputs)

        
        return mi_lb