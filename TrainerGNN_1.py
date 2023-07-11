#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/10 10:28
# @Author  : zhixiuma
# @File    : TrainerGNN_1.py
# @Project : Test
# @Software: PyCharm


from load_dataset.load_data import Dataset
from trainer_args import parse_args
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import time
import math
import torch.nn.functional as F
from utils.utils import judge_dir,seed_everything
from model.classifier import MLP_classifier
from model.gnn import GCN,SAGE,GIN,GAT,MLP
from utils.utils import evaluate_1,evaluate
import os

SAGE_lr_wd =  {
        'encoder':{
            'bail': 0.01,
            'credit': 0.001,
            'pokec_n': 0.001,
            'pokec_z': 0.001,

        },
        'classifier':{
            'bail': 0.001,
            'credit': 0.001,
            'pokec_n': 0.001,
            'pokec_z': 0.001,

        }
    }


GIN_lr_wd = {
        'encoder':{
            'bail': 0.01,
            'credit': 0.01,
            'pokec_n': 0.001,
            'pokec_z': 0.01,

        },
        'classifier':{
            'bail': 0.01,
            'credit': 0.01,
            'pokec_n': 0.001,
            'pokec_z': 0.01,

        }
    }

GCN_lr_wd = {
        'encoder':{
            'bail': 0.01,
            'credit': 0.001,
            'pokec_n': 0.01,
            'pokec_z': 0.01,

        },
        'classifier':{
            'bail': 0.01,
            'credit': 0.001,
            'pokec_n': 0.01,
            'pokec_z': 0.01,

        },
    }

GAT_lr_wd = {
        'encoder':{
            'bail': 0.01,
            'credit': 0.001,
            'pokec_n': 0.001,
            'pokec_z': 0.001,

        },
        'classifier':{
            'bail': 0.01,
            'credit': 0.001,
            'pokec_n': 0.001,
            'pokec_z': 0.001,
        },
    }



def lr_wd_specify(args):

    if args.gnn=='SAGE':
        args.e_lr = SAGE_lr_wd.get('encoder').get(args.dataset)
        args.c_lr = SAGE_lr_wd.get('classifier').get(args.dataset)

    if args.gnn=='GIN':
        args.e_lr = SAGE_lr_wd.get('encoder').get(args.dataset)
        args.c_lr = SAGE_lr_wd.get('classifier').get(args.dataset)

    if args.gnn=='GCN':
        args.e_lr = GCN_lr_wd.get('encoder').get(args.dataset)
        args.c_lr = GCN_lr_wd.get('classifier').get(args.dataset)

    if args.gnn=='GAT':
        args.e_lr = GAT_lr_wd.get('encoder').get(args.dataset)
        args.c_lr = GAT_lr_wd.get('classifier').get(args.dataset)


    return args

def main_1(pyg_data,args):
    save_dir = '/home/mzx/GNNDebias_1/Original_Results_2023_6_17_(2)/'+args.gnn

    pbar = tqdm(range(args.runs), unit='run')
    acc, f1, auc_roc, parity, equality, times = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)
    #
    # args = lr_wd_specify(args)
    # print(args.e_lr)
    # print(args.c_lr)
    # print(args.e_wd)

    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = torch.optim.Adam([
        dict(params=classifier.lin.parameters(), weight_decay=args.c_wd)], lr=args.c_lr)

    # print(args)
    if (args.gnn == 'GCN'):
        gnn = GCN(args).to(args.device)
        optimizer_e = torch.optim.Adam(gnn.parameters(), lr=args.e_lr)
    elif (args.gnn == 'GIN'):
        gnn = GIN(args).to(args.device)
        optimizer_e = torch.optim.Adam(gnn.parameters(), lr=args.e_lr)
    elif (args.gnn == 'SAGE'):
        gnn = SAGE(args).to(args.device)
        optimizer_e = torch.optim.Adam(gnn.parameters(), lr=args.e_lr)
    elif (args.gnn == 'GAT'):
        gnn = GAT(args).to(args.device)
        optimizer_e = torch.optim.Adam(gnn.parameters(), lr=args.e_lr)
    elif (args.gnn == 'MLP'):
        gnn = MLP(args).to(args.device)
        optimizer_e = torch.optim.Adam(gnn.parameters(), lr=args.e_lr)

    for count in pbar:
        seed_everything(count + args.seed)
        gnn.reset_parameters()
        classifier.reset_parameters()

        best_val_tradeoff = 0
        best_val_loss = math.inf
        start = time.time()
        for epoch in range(0, args.epochs):

            gnn.train()
            classifier.train()

            optimizer_e.zero_grad()
            optimizer_c.zero_grad()

            output_1 = gnn(pyg_data.x, pyg_data.edge_index)
            output = classifier(output_1)

            loss_c = F.binary_cross_entropy_with_logits(
                output[pyg_data.train_mask].float(), pyg_data.y[pyg_data.train_mask].unsqueeze(1).float().to(args.device))
            loss_c.backward()

            optimizer_e.step()
            optimizer_c.step()

            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_1(classifier,gnn,pyg_data, args)

            if auc_rocs['val']+accs['val']+ F1s['val']-(tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:

                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

                embeddings = output_1
                best_val_tradeoff = accs['val']

                ckpt = {
                    'gnn_model_state': gnn.state_dict(),
                    'classfier_model_state': classifier.state_dict(),
                    'optimizer_e_state': optimizer_e.state_dict(),
                    'optimizer_c_state': optimizer_c.state_dict(),
                }
                judge_dir(args.checkpoint_dir)
                torch.save(ckpt, os.path.join(args.checkpoint_dir,args.gnn+'_'+args.dataset+'_model_best.pt'))

        print(args.e_lr)
        print(args.c_lr)
        print(args.e_wd)

        acc[count] = test_acc
        f1[count] = test_f1
        auc_roc[count] = test_auc_roc
        parity[count] = test_parity
        equality[count] = test_equality
        times[count] = time.time() - start

        save_dir_1 = save_dir+'/embeddings/'
        judge_dir(save_dir_1)
        np.savetxt(save_dir_1 + args.encoder + '_' + args.dataset + '_' + str(count) + '.txt',
                   np.array(embeddings.detach().cpu().numpy()))

    return acc, f1, auc_roc, parity, equality, times

if __name__=='__main__':
    import warnings
    warnings.filterwarnings('ignore')
    args = parse_args()

    for gnn in ['GCN']: #
        args.gnn = gnn
        for dataset in ['pokec_n','pokec_z','bail','credit']:# 'bail','bail','german','german'
            args.dataset = dataset
            if args.dataset=='german':
                sens_attr = "Gender"
                predict_attr = "GoodCustomer"
                path = r"/home/mzx/GNNDebias_1/dataset/german/"
                label_number=1000
            elif args.dataset=='bail':
                sens_attr="WHITE"
                predict_attr="RECID"
                path=r"/home/mzx/GNNDebias_1/dataset/bail/"
                label_number=1000
            elif args.dataset=='credit':
                sens_attr = 'Age'
                predict_attr = 'NoDefaultNextMonth'
                path = r"/home/mzx/GNNDebias_1/dataset/credit/"
                label_number = 100
            elif args.dataset != 'nba':
                sens_attr = "region"
                predict_attr = "I_am_working_in_field"
                label_number = 500
                seed = 20
                path = "/home/mzx/GNNDebias_1/dataset/pokec/"
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

            print('**********************' + args.dataset + args.gnn+'**********************')
            data = Dataset(name=args.dataset, sens_attr=sens_attr, predict_attr=predict_attr,
                           path=path, label_number=label_number)
            print(data)

            pyg_data = data.pyg_data.to(device=args.device)
            args.num_classes = 1
            args.num_features = data.features.shape[1]

            acc, f1, auc_roc, parity, equality, times = main_1(pyg_data,args)
            result_save = []
            var_save = []
            print('======' + args.dataset + args.gnn + '======')
            print('auc_roc:', np.mean(auc_roc) * 100, np.std(auc_roc) * 100)
            print('Acc:', np.mean(acc) * 100, np.std(acc) * 100)
            print('f1:', np.mean(f1) * 100, np.std(f1) * 100)
            print('parity:', np.mean(parity) * 100, np.std(parity) * 100)
            print('equality:', np.mean(equality) * 100, np.std(equality) * 100)
            print('times:', np.mean(times) * 100, np.std(times) * 100)
            result_save.append(np.mean(auc_roc))
            result_save.append(np.mean(acc))
            result_save.append(np.mean(f1))
            result_save.append(np.mean(parity))
            result_save.append(np.mean(equality))

            var_save.append(np.std(auc_roc))
            var_save.append(np.std(acc))
            var_save.append(np.std(f1))
            var_save.append(np.std(parity))
            var_save.append(np.std(equality))

            save_dir = '/home/mzx/GNNDebias_1/Original_Results_2023_6_17_(2)/'+args.gnn
            from utils.utils import judge_dir
            judge_dir(save_dir)
            print(save_dir)
            np.savetxt("{}/{}_{}_res.txt".format(save_dir, args.dataset, 'GNNDebias_1'), np.array(result_save))
            np.savetxt("{}/{}_{}_val.txt".format(save_dir, args.dataset, 'GNNDebias_1'), np.array(var_save))

