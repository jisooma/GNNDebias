#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 21:45
# @Author  : zhixiuma
# @File    : gnn.py
# @Project : Test
# @Software: PyCharm
import torch
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn as nn
from torch_geometric.nn.conv import SAGEConv, GINConv, GCNConv, GATConv
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter
from torch.nn import functional as F


class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args

        self.lin = Linear(args.num_features, args.hidden)
        # self.dropout = nn.Dropout(args.dropout)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x,edge_index,return_all_emb=False):
        h = self.lin(x)
        # h = F.relu(h)
        # h = self.dropout(h)
        if return_all_emb:
            return h,h
        return h


class GCN(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        self.conv1 = GCNConv(args.num_features, args.hidden)
        # self.conv2 = GCNConv(args.hidden, args.hidden)
        self.dropout = nn.Dropout(args.dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        # self.conv2.reset_parameters()


    def forward(self, x, edge_index, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        # x2 = self.conv2(x1,edge_index)

        if return_all_emb:
            return x1, x1

        return x1


class GIN(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(args.num_features, args.hidden),nn.BatchNorm1d(args.hidden)))
        self.dropout = nn.Dropout(args.dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, x, edge_index, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)

        if return_all_emb:
            return x1, x1
        return x1


class SAGE(nn.Module):
    def __init__(self, args):
        super(SAGE, self).__init__()
        self.args = args
        self.conv1 = SAGEConv(args.num_features, args.hidden, normalize=True)
        self.conv1.aggr = 'mean'
        self.dropout = nn.Dropout(args.dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, x, edge_index,return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)

        if return_all_emb:
            return x1, x1
        return x1


import torch
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.conv1 = GATConv(args.num_features, args.hidden)
        # self.conv2 = GATConv(nhid,nhid)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, x, edge_index,return_all_emb=False):

        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, training=self.training)
        # x2 = self.conv2(x1, edge_index)
        if return_all_emb:
            return x1, x1

        return x1
