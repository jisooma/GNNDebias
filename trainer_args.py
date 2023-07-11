#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/9 21:18
# @Author  : zhixiuma
# @File    : trainer_args.py
# @Project : Test
# @Software: PyCharm
import argparse

import torch

def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--Debiasing_model', type=str, default='',
                        help='Debiasing method')
    parser.add_argument('--gnn', type=str, default='mlp',
                        help='GNN architecture')
    parser.add_argument('--num_nodes', type=int, default=0,
                        help='number nodes')

    parser.add_argument('--acc', type=float, default=0.90,
                        help='the selected FairGNN accuracy on val would be at least this high')
    parser.add_argument('--roc', type=float, default=0.93,
                        help='the selected FairGNN ROC score on val would be at least this high')

    # Training
    # parser.add_argument('--lr', type=float, default=1e-3,
    #                     help='initial learning rate')
    # parser.add_argument('--weight_decay', type=float, default=1e-5,
    #                     help='weight decay')

    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)


    # # GAT
    # # debias
    # parser.add_argument('--d_epochs', type=int, default=5)
    # parser.add_argument('--g_epochs', type=int, default=5)
    # parser.add_argument('--c_epochs', type=int, default=5)
    # parser.add_argument('--c_lr', type=float, default=0.01)
    # parser.add_argument('--c_wd', type=float, default=0)
    # parser.add_argument('--e_lr', type=float, default=0.01)
    # parser.add_argument('--e_wd', type=float, default=0)
    # parser.add_argument('--d_lr', type=float, default=0.01)
    # parser.add_argument('--d_wd', type=float, default=0)

    #train original model
    # parser.add_argument('--c_lr', type=float, default=0.01)
    # parser.add_argument('--c_wd', type=float, default=1e-5)
    # parser.add_argument('--e_lr', type=float, default=0.01)
    # parser.add_argument('--e_wd', type=float, default=1e-5)




    # GCN
    parser.add_argument('--c_lr', type=float, default=0.001)
    parser.add_argument('--c_wd', type=float, default=0)
    parser.add_argument('--e_lr', type=float, default=0.01)
    parser.add_argument('--e_wd', type=float, default=0)

    parser.add_argument('--d_lr', type=float, default=0.01)
    parser.add_argument('--d_wd', type=float, default=0)
    parser.add_argument('--d_epochs', type=int, default=5)
    parser.add_argument('--g_epochs', type=int, default=5)
    parser.add_argument('--c_epochs', type=int, default=5)
    #
    # parser.add_argument('--c_lr', type=float, default=0.001)
    # parser.add_argument('--c_wd', type=float, default=0)
    # parser.add_argument('--e_lr', type=float, default=0.001)
    # parser.add_argument('--e_wd', type=float, default=0)
    #


    # # # SAGE
    # parser.add_argument('--d_epochs', type=int, default=5)
    # parser.add_argument('--g_epochs', type=int, default=5)
    # parser.add_argument('--c_epochs', type=int, default=5)
    #
    # # # pokec_z:
    # # parser.add_argument('--d_epochs', type=int, default=5)
    # # parser.add_argument('--c_lr', type=float, default=0.001)
    # # parser.add_argument('--c_wd', type=float, default=0)
    # # parser.add_argument('--e_lr', type=float, default=0.001)
    # # parser.add_argument('--e_wd', type=float, default=0)
    #
    # parser.add_argument('--d_lr', type=float, default=0.001)
    # parser.add_argument('--d_wd', type=float, default=0)
    # # sage original
    # parser.add_argument('--c_lr', type=float, default=0.001)
    # parser.add_argument('--c_wd', type=float, default=1e-5)
    # parser.add_argument('--e_lr', type=float, default=0.001)
    # parser.add_argument('--e_wd', type=float, default=1e-5)


    # # # # GIN
    # parser.add_argument('--d_epochs', type=int, default=5)
    # parser.add_argument('--g_epochs', type=int, default=10)
    # parser.add_argument('--c_epochs', type=int, default=10)
    #
    # parser.add_argument('--c_lr', type=float, default=0.01)
    # parser.add_argument('--c_wd', type=float, default=0)
    # parser.add_argument('--e_lr', type=float, default=0.01)
    # parser.add_argument('--e_wd', type=float, default=0)
    #
    # parser.add_argument('--d_lr', type=float, default=0.01)
    # parser.add_argument('--d_wd', type=float, default=0)



    parser.add_argument('--beta', type=float, default=0.01,
                        help='The hyperparameter of beta')
    # parser.add_argument('--early_stopping', type=int, default=0)
    # parser.add_argument('--prop', type=str, default='scatter')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--encoder', type=str, default='GCN')
    # parser.add_argument('--clip_e', type=float, default=1)
    # parser.add_argument('--f_mask', type=str, default='yes')
    # parser.add_argument('--weight_clip', type=str, default='yes')
    # parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    # parser.add_argument('--K', type=int, default=10)

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_dir_2023_6_15/original/')
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return args
"""
(1)0.001,0.001
(2)0.001,0.01
(3)0.01,0.001
(4)0.01,0.01
"""