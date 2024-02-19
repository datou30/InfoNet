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

    joint = batch[:,:,[0,1]]
    marginal = batch[:,:,[2,3]]
    #reshape_merge = torch.reshape(merge, (1,200,100))
    #j_matrix = model(joint)  
    #m_matrix = model(marginal)
    #print('===================> matrix range is:', torch.max(j_matrix), torch.min(j_matrix))
    
    t = lookup_value_grid(matrix, joint, "bilinear")
    #print('===================> t range is:', torch.max(t), torch.min(t) ,'\n')
    et = torch.exp(lookup_value_grid(matrix, marginal, "bilinear"))
    #print('===================> et range is:', torch.max(et), torch.min(et),'\n')
    #t_m = lookup_value_grid(matrix, marginal, "bilinear")
    if len(t.shape)==1:
        t = t.unsqueeze(0)
        et = et.unsqueeze(0)
    mi_lb = torch.mean(t, dim=1) - torch.log(torch.mean(et, dim=1))
    if torch.isnan(- torch.mean(mi_lb) ):
        raise ValueError("Loss is NaN!")
    return mi_lb

class LapConv(nn.Module):
    def __init__(self, padding='same'):
        super(LapConv, self).__init__()
        self.padding = padding
        self.kernel = torch.nn.Parameter(torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3), requires_grad=False)

    def forward(self, img):
        c_i = img.shape[1]
        
        if self.padding == 'same':
            padding = self.kernel.shape[2] // 2
        else:
            padding = 0
        return F.conv2d(img, self.kernel, padding=padding, stride=1, groups=c_i)

class PerceiverIO_lap(nn.Module):

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
        
        self.lap = LapConv()
        
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
        torch.save(outputs.cpu().numpy(), "lookuptable.pth")
        laps_value = self.lap(outputs.unsqueeze(1))
        lap_sum = torch.mean(torch.abs(laps_value))
        torch.save(outputs.cpu().numpy(), "lookuptable_smoothed.pth")
        print("saved!!!!!!!!!!!!!!!1")
        mi_lb = mutual_information(inputs, outputs)

        
        return mi_lb, lap_sum