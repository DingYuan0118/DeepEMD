from os import truncate
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .emd_utils import *
from Models.models.mmd_utils import gaussian_kernel, mmd
import timm
from einops import rearrange
import numpy as np
    
def get_mmd_distance(proto, query):
    """
    计算query与proto之间的MMD距离
        Args:
            proto(tensor):shape [n_way, dim, patch, patch] eg [5,640,5,5]
            query(tensor):shape [n_query, dim, patch, path] eg [75,640,5,5]
        
        Return:
            logits(tensor):shape [n_query, n_way]
    """
    num_query = query.shape[0]
    num_proto = proto.shape[0]
    proto = rearrange(proto, 'b dim rows cols -> b (rows cols) dim')
    query = rearrange(query, 'b dim rows cols -> b (rows cols) dim')
    # init score函数
    score = torch.randn(num_query, num_proto) # shape eg.[75,5]
    for i in range(num_query):
        for j in range(num_proto):
            score[i][j] = mmd(query[i], proto[j])
    logits = - score
    return logits


data_1 = torch.tensor(np.random.normal(loc=0,scale=10,size=(25,640)), requires_grad=True)
data_2 = torch.tensor(np.random.normal(loc=10,scale=10,size=(25,640)), requires_grad=True)
data_3 = torch.tensor(np.random.normal(loc=20,scale=10,size=(25,640)), requires_grad=True)
data_4 = torch.tensor(np.random.normal(loc=30,scale=10,size=(25,640)), requires_grad=True)
data_5 = torch.tensor(np.random.normal(loc=40,scale=10,size=(25,640)), requires_grad=True)
proto = torch.stack((data_1,data_2,data_3,data_4,data_5))
proto = rearrange(proto, 'b (rows cols) dim -> b dim rows cols', rows=5)
query = torch.tensor(np.random.normal(loc=15,scale=10,size=(75,640,5,5)), requires_grad=True)
label = torch.tensor(np.array((1)).repeat(75,axis=0))



logits = get_mmd_distance(proto, query)
print(logits.shape)
loss = F.cross_entropy(logits, label)
print(loss)

