#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 17:19
# @Author  : zhixiuma
# @File    : encoder.py
# @Project : Test
# @Software: PyCharm

import torch
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn as nn
from torch_geometric.nn.conv import SAGEConv,GINConv,GCNConv,GATConv
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter
from torch.nn import functional as F

class GCN(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        self.conv1 = GCNConv(args.in_dim, args.hidden_dim)
        self.conv2 = GCNConv(args.hidden_dim, args.out_dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, edge_index, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x = F.relu(x1)
        x = self.dropout(x)
        x2 = self.conv2(x, edge_index)

        if return_all_emb:
            return x1, x2

        return x2


class GAT(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.conv1 = GATConv(args.in_dim, args.hidden_dim)
        self.conv2 = GATConv(args.hidden_dim, args.out_dim)

    def forward(self, x, edge_index, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x = F.relu(x1)
        x2 = self.conv2(x, edge_index)

        if return_all_emb:
            return x1, x2

        return x2


class GIN(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.conv1 = GINConv(nn.Linear(args.in_dim, args.hidden_dim))
        self.conv2 = GINConv(nn.Linear(args.hidden_dim, args.out_dim))

    def forward(self, x, edge_index, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x = F.relu(x1)
        x2 = self.conv2(x, edge_index)
        if return_all_emb:
            return x1, x2
        return x2



class SAGE(nn.Module):
    def __init__(self, args):
        super(SAGE, self).__init__()

        self.args = args

        self.conv1 = SAGEConv(args.num_features, args.hidden, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Dropout(p=args.dropout)
        )
        self.conv2 = SAGEConv(args.hidden, args.hidden, normalize=True)
        self.conv2.aggr = 'mean'

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index,return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x = self.transition(x)
        x2 = self.conv2(x, edge_index)
        if return_all_emb:
            return x1, x2
        return x2




class GCN_encoder(torch.nn.Module):
    def __init__(self,nfeat,nhid,nclass,dropout):
        super().__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid,nclass)

    def forward(self, x,edge_index):
        x = self.body(x,edge_index)
        x = self.fc(x)
        # return F.log_softmax(x, dim=1)
        return x

class GCN_Body(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.conv1 = GCNConv(args.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden,args.hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


    def forward(self, x,edge_index):
        # x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        return x
"""
encodder
"""
class MLP_encoder(torch.nn.Module):
    def __init__(self, args):
        super(MLP_encoder, self).__init__()
        self.args = args

        self.lin = Linear(args.num_features, args.hidden)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

    def forward(self, x, edge_index=None, mask_node=None):
        h = self.lin(x)

        return h

def propagate2(x, edge_index):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(
        edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


class GCN_encoder_scatter(torch.nn.Module):
    def __init__(self, args):
        super(GCN_encoder_scatter, self).__init__()

        self.args = args

        self.lin = Linear(args.num_features, args.hidden, bias=False)

        self.bias = Parameter(torch.Tensor(args.hidden))

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):

            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h = self.lin(x)
        h = propagate2(h, edge_index) + self.bias
        return h


class GIN_encoder(nn.Module):
    def __init__(self, args):
        super(GIN_encoder, self).__init__()

        self.args = args

        self.mlp = nn.Sequential(
            nn.Linear(args.num_features, args.hidden),
            # nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            # nn.Linear(args.hidden, args.hidden),
        )

        self.conv = GINConv(self.mlp)

    def clip_parameters(self, channel_weights):
        for i in range(self.mlp[0].weight.data.shape[1]):
            self.mlp[0].weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                      self.args.clip_e * channel_weights[i])

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        return h


class SAGE_encoder(nn.Module):
    def __init__(self, args):
        super(SAGE_encoder, self).__init__()

        self.args = args

        self.conv1 = SAGEConv(args.num_features, args.hidden, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Dropout(p=args.dropout)
        )
        self.conv2 = SAGEConv(args.hidden, args.hidden, normalize=True)
        self.conv2.aggr = 'mean'

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def clip_parameters(self, channel_weights):
        for i in range(self.conv1.lin_l.weight.data.shape[1]):
            self.conv1.lin_l.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                           self.args.clip_e * channel_weights[i])

        for i in range(self.conv1.lin_r.weight.data.shape[1]):
            self.conv1.lin_r.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                           self.args.clip_e * channel_weights[i])

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        h = self.conv2(x, edge_index)
        return h