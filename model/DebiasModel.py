#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/9 19:47
# @Author  : zhixiuma
# @File    : DebiasModel.py
# @Project : Test
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gnn import GCN,SAGE,GIN,GAT
from torch.nn.utils import spectral_norm

class DebiasLayer(nn.Module):
    def __init__(self, args):
        super(DebiasLayer, self).__init__()
        self.weights = nn.Parameter(torch.ones(args.hidden, args.hidden) / 1000)

    def reset_parameters(self):
        self.weights = torch.nn.init.xavier_uniform_(self.weights)

    def forward(self,x):
        return torch.matmul(x,self.weights)


class DebiasLayer_1(nn.Module):
    def __init__(self, h1,h2):
        super(DebiasLayer_1, self).__init__()
        self.weights = nn.Parameter(torch.ones(h1,h2) / 1000)
        self.reset_parameters()

    def reset_parameters(self):
        self.weights = torch.nn.init.xavier_uniform_(self.weights)

    def forward(self,x):
        return torch.matmul(x,self.weights)


class DebiasGCN(GCN):
    def __init__(self, args,**kwargs):
        super().__init__(args)
        # self.debiaslayer1 = DebiasLayer_1(args.hidden,args.hidden)
        self.debiaslayer2 = DebiasLayer_1(args.hidden,args.hidden)
        self.conv1.requires_grad = False
        # self.conv2.requires_grad = False

    def reset_parameters(self):
        self.debiaslayer2.reset_parameters()

    def forward(self, x, edge_index, return_all_emb=False):

        x1 = self.conv1(x, edge_index)
        # x1 = self.debiaslayer1(x1)
        # x = F.relu(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        # x2 = self.conv2(x, edge_index)
        x2 = self.debiaslayer2(x1)

        if return_all_emb:
            return x1, x2
        return x2

    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)



class DebiasGIN(GIN):
    def __init__(self, args,  **kwargs):
        super().__init__(args)
        self.debiaslayer2 = DebiasLayer_1(args.hidden,args.hidden)
        self.conv1.requires_grad = False

    def reset_parameters(self):
        self.debiaslayer2.reset_parameters()

    def forward(self, x, edge_index,  return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x = F.relu(x1)
        x = self.dropout(x)
        x2 = self.debiaslayer2(x)

        if return_all_emb:
            return x1, x2

        return x2

    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)



class DebiasSAGE(SAGE):
    def __init__(self, args, **kwargs):
        super().__init__(args)
        # self.debiaslayer1 = DebiasLayer_1(args.hidden,args.hidden)
        self.debiaslayer2 = DebiasLayer_1(args.hidden,args.hidden)

        self.conv1.requires_grad = False

    def reset_parameters(self):
        self.debiaslayer2.reset_parameters()

    def forward(self, x, edge_index, return_all_emb=False):

        x1 = self.conv1(x, edge_index)
        x = F.relu(x1)
        x1 = self.dropout(x)
        x2 = self.debiaslayer2(x1)

        if return_all_emb:
            return x1, x2

        return x2

    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)


class DebiasGAT(GAT):
    def __init__(self, args, **kwargs):
        super().__init__(args)
        # self.debiaslayer1 = DebiasLayer_1(args.hidden,args.hidden)
        self.debiaslayer2 = DebiasLayer_1(args.hidden,args.hidden)

        self.conv1.requires_grad = False
        # self.conv2.requires_grad = False

    def forward(self, x, edge_index, return_all_emb=False):

        x1 = self.conv1(x, edge_index)
        x = F.relu(x1)
        # x = self.
        x1 = self.debiaslayer2(x)
        # x2 = self.conv2(x,edge_index)
        # x2 = self.debiaslayer2(x2)

        if return_all_emb:
            return x1, x1

        return x1

    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)