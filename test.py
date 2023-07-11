#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/17 16:25
# @Author  : zhixiuma
# @File    : test.py
# @Project : GNNDebias_1
# @Software: PyCharm
import math

import torch
from torch.nn import Linear
from model.gnn import  GCN,GIN,SAGE,MLP
import os
from tqdm import tqdm

class MLP_discriminator(torch.nn.Module):
    def __init__(self, args):
        super(MLP_discriminator, self).__init__()
        self.args = args
        self.lin = Linear(args.hidden, 1)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h):
        h = self.lin(h)
        return torch.sigmoid(h)

from torch import nn
# from deeprobust.
def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
    # print(preds)
    # print(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    # print(correct)
    return correct / len(labels)

def run(data,args):
    acc_1 = np.zeros(args.runs),
    print(acc_1)
    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = torch.optim.Adam([
        dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)], lr=args.d_lr)

    if (args.encoder == 'GCN'):
        encoder = GCN(args).to(args.device)
        optimizer_e = torch.optim.Adam(encoder.parameters(), lr=args.e_lr)
    elif (args.encoder == 'GIN'):
        encoder = GIN(args).to(args.device)
        optimizer_e = torch.optim.Adam(encoder.parameters(), lr=args.e_lr)
    elif (args.encoder == 'SAGE'):
        encoder = SAGE(args).to(args.device)
        optimizer_e = torch.optim.Adam(encoder.parameters(), lr=args.e_lr)
    elif (args.encoder == 'MLP'):
        encoder = MLP(args).to(args.device)
        optimizer_e = torch.optim.Adam(encoder.parameters(), lr=args.e_lr)


    criterion = nn.BCELoss()
    loss_best = math.inf

    pbar = tqdm(range(args.runs), unit='run')

    for count in pbar:
        best_acc = 0
        for epoch_d in range(0, args.d_epochs):
            model_ckpt = torch.load(os.path.join(r'/home/mzx/GNNDebias_1/checkpoint_dir_2023_6_15/original/',
                                                 args.encoder + '_' + args.dataset + '_' + 'model_best.pt'),
                                    map_location=args.device)
            encoder.load_state_dict(model_ckpt['gnn_model_state'], strict=False)

            optimizer_d.zero_grad()
            # 尽可能识别出原始的敏感属性
            h1, h2 = encoder(data.x, data.edge_index, return_all_emb=True)
            output2 = discriminator(h2)
            loss_d = criterion(output2.view(-1),
                               data.x[:, data.sens_idx])
            acc_train = accuracy(output2, data.x[:, data.sens_idx])
            loss_d.backward()
            optimizer_d.step()

            if acc_train > best_acc:
                best_acc = acc_train
                # torch.save(model.state_dict(), "./checkpoint/GCN_sens_{}_ns_{}".format(dataset, sens_number))
        print(best_acc)
        # acc_1[count] = best_acc.detach().cpu().numpy()

    return acc_1



if __name__=='__main__':
    import warnings
    from load_dataset.load_data import Dataset
    from trainer_args import parse_args
    warnings.filterwarnings('ignore')
    args = parse_args()
    import numpy as np
    for encoder in ['MLP','GCN','SAGE','GIN']:  # 'MLP','GCN','GIN''german','credit'
        args.encoder = encoder
        for dataset in ['bail','credit', ]:  # 'bail',,'GCN','GIN''bail','credit'
            args.dataset = dataset
            if args.dataset == 'german':
                sens_attr = "Gender"
                predict_attr = "GoodCustomer"
                path = r"/home/mzx/GNNDebias_1/dataset/german/"
                label_number = 1000
            elif args.dataset == 'bail':
                sens_attr = "WHITE"
                predict_attr = "RECID"
                path = r"/home/mzx/GNNDebias_1/dataset/bail/"
                label_number = 1000
            elif args.dataset == 'credit':
                sens_attr = 'Age'
                predict_attr = 'NoDefaultNextMonth'
                path = r"/home/mzx/GNNDebias_1/dataset/credit/"
                label_number = 100
            elif args.dataset != 'nba':
                if args.dataset == 'pokec-z':
                    dataset = 'region_job'
                else:
                    dataset = 'region_job_2'
                sens_attr = "region"
                predict_attr = "I_am_working_in_field"
                label_number = 500
                path = r"/home/mzx/GNNDebias_1/dataset/pokec/"
                test_idx = False
            else:
                dataset = 'nba'
                sens_attr = "country"
                predict_attr = "SALARY"
                label_number = 100
                sens_number = 50
                seed = 20
                path = "../dataset/NBA"
                test_idx = True

            print('**********************' + args.dataset + '**********************')
            data = Dataset(name=args.dataset, sens_attr=sens_attr, predict_attr=predict_attr,
                           path=path, label_number=label_number)
            print(data)

            pyg_data = data.pyg_data.to(device=args.device)
            args.num_classes = 1
            print(args.num_classes)
            args.num_features = data.features.shape[1]

            acc = run(pyg_data, args)
            result_save = []
            var_save = []
            print('======' + args.dataset + args.encoder + '======')

            print('Acc:', np.mean(acc) * 100, np.std(acc) * 100)
            var_save.append(np.std(acc))

            #
            # save_dir = '/home/mzx/GNNDebias_1/Debias_Results_2023_6_15/Encoder/' + args.encoder
            # from utils.utils import judge_dir
            #
            # judge_dir(save_dir)
            # np.savetxt("{}/{}_{}_res.txt".format(save_dir, args.dataset, 'GNNDebias_1'), np.array(result_save))
            # np.savetxt("{}/{}_{}_val.txt".format(save_dir, args.dataset, 'GNNDebias_1'), np.array(var_save))

