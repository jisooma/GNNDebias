#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 22:09
# @Author  : zhixiuma
# @File    : TrainerGNNDebias_1.py
# @Project : Test
# @Software: PyCharm

class Trainer():
    def __init__(self):
        pass

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
from model.DebiasModel import DebiasGCN,DebiasGIN,DebiasSAGE,DebiasGAT
from model.discriminator import MLP_discriminator
# from
from utils.utils import evaluate_4,evaluate_3,evaluate_2,evaluate_1
import os

SAGE_lr_wd =  {
        'encoder':{
            'bail': 0.01,
            'pokec_n': 0.001,
            'pokec_z': 0.001,
            'credit': 0.001,
        },
        'classifier':{
            'bail': 0.001,
            'pokec_n': 0.001,
            'pokec_z': 0.001,
            'credit': 0.001,
        },
        'discriminator':{
            'bail': 0.01,
            'pokec_n': 0.001,
            'pokec_z': 0.001,
            'credit': 0.001,
        },
        'beta':{
                'bail': 0.1,
                'credit': 0.001,
                'pokec_n': 0.1,
                'pokec_z': 0.01,
            },
        'alpha':{
                'bail': 0,
                'credit': 0.1,
                'pokec_n': 0,
                'pokec_z': 1,
            }
    }


GIN_lr_wd =  {
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

        },
    'discriminator': {
        'bail': 0.001,
        'pokec_n': 0.001,
        'pokec_z': 0.001,
        'credit': 0.001,
    },
    'beta': {
        'bail': 0.1,
        'credit': 0.01,
        'pokec_n': 0.15,
        'pokec_z': 0.01,
    },
    'alpha': {
        'bail': 0.0001,
        'credit': 0.1,
        'pokec_n': 1,
        'pokec_z': 0.1,
    }
    }


GCN_lr_wd =  {
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
        'discriminator': {
            'bail': 0.001,
            'credit': 0.001,
            'pokec_n': 0.01,
            'pokec_z': 0.001,
        },
        'beta': {
            'bail': 1,
            'credit': 0.01,
            'pokec_n': 0.01,
            'pokec_z': 0.001,
        },
        'alpha': {
            'bail': 1,
            'credit': 1,
            'pokec_n': 1,
            'pokec_z': 1,
        }
    }

GAT_lr_wd = {
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

        },
        'discriminator': {
            'bail': 0.001,
            'pokec_n': 0.001,
            'pokec_z': 0.001,
            'credit': 0.001,
        },
        'beta': {
            'bail': 1,
            'credit': 0.01,
            'pokec_n': 0.001,
            'pokec_z': 0.01,
        },
        'alpha': {
            'bail': 0.001,
            'credit': 1,
            'pokec_n': 1,
            'pokec_z': 1,
        }
    }


def lr_wd_specify(args):
    if args.encoder=='SAGE':
        args.e_lr = SAGE_lr_wd.get('encoder').get(args.dataset)
        args.c_lr = SAGE_lr_wd.get('classifier').get(args.dataset)
        args.beta = SAGE_lr_wd.get('beta').get(args.dataset)
        args.alpha = SAGE_lr_wd.get('alpha').get(args.dataset)

    if args.encoder=='GIN':
        args.e_lr = GIN_lr_wd.get('encoder').get(args.dataset)
        args.c_lr = GIN_lr_wd.get('classifier').get(args.dataset)
        args.beta = GIN_lr_wd.get('beta').get(args.dataset)
        args.alpha = GIN_lr_wd.get('alpha').get(args.dataset)

    if args.encoder=='GCN':
        args.e_lr = GCN_lr_wd.get('encoder').get(args.dataset)
        args.c_lr = GCN_lr_wd.get('classifier').get(args.dataset)
        args.beta = GCN_lr_wd.get('beta').get(args.dataset)
        args.alpha = GCN_lr_wd.get('alpha').get(args.dataset)

    if args.encoder=='GAT':
        args.e_lr = GAT_lr_wd.get('encoder').get(args.dataset)
        args.c_lr = GAT_lr_wd.get('classifier').get(args.dataset)
        args.beta = GAT_lr_wd.get('beta').get(args.dataset)
        args.alpha = GAT_lr_wd.get('alpha').get(args.dataset)

    return args


def main_3(data,args):
    save_dir = '/home/mzx/GNNDebias_1/Debias_Results_2023_6_15/' + args.encoder

    pbar = tqdm(range(args.runs), unit='run')
    criterion = nn.BCELoss()
    acc, f1, auc_roc, parity, equality, times = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    print(args.encoder)
    args = lr_wd_specify(args)
    print("****************beta")
    print(args.beta)
    print(args.alpha)

    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = torch.optim.Adam([
        dict(params=classifier.lin.parameters(), weight_decay=args.c_wd)], lr=args.c_lr)

    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = torch.optim.Adam([
        dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)], lr=args.d_lr)


    if (args.encoder == 'GCN'):
        encoder = DebiasGCN(args).to(args.device)
    elif (args.encoder == 'GIN'):
        encoder = DebiasGIN(args).to(args.device)
    elif (args.encoder == 'SAGE'):
        encoder = DebiasSAGE(args).to(args.device)
    elif (args.encoder == 'GAT'):
        encoder = DebiasGAT(args).to(args.device)

    parameters_to_optimize = [
        {'params': [p for n, p in encoder.named_parameters() if 'debiaslayer' in n], 'weight_decay': 0.0}
    ]

    optimizer_e = torch.optim.Adam(parameters_to_optimize, lr=args.e_lr)
    print('parameters_to_optimize', [n for n, p in encoder.named_parameters() if 'debiaslayer' in n])

    for count in pbar:
        model_ckpt = torch.load(os.path.join(r'/home/mzx/GNNDebias_1/checkpoint_dir_2023_6_15/original/',
                                             args.encoder + '_' + args.dataset + '_' + 'model_best.pt'),
                                map_location=args.device)
        encoder.load_state_dict(model_ckpt['gnn_model_state'], strict=False)
        classifier.load_state_dict(model_ckpt['classfier_model_state'], strict=False)

        accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_2(classifier, encoder, data, args)

        print(
            'acc_test: {:.4f}'.format(accs['test']),
            "roc_test: {:.4f}".format(auc_rocs['test']),
            "parity_test: {:.4f}".format(tmp_parity['test']),
            "equality_test: {:.4f}".format(tmp_equality['test']))


        seed_everything(count + args.seed)
        discriminator.reset_parameters()
        classifier.reset_parameters()

        best_val_tradeoff = 0
        best_val_loss = math.inf
        start = time.time()
        best_debias_weights = 0
        for epoch in range(0, args.epochs):
            # train discriminator to recognize the sensitive group
            discriminator.train()
            encoder.eval()
            for epoch_d in range(0, args.d_epochs):
                optimizer_d.zero_grad()
                # 尽可能识别出原始的敏感属性
                h1, h2 = encoder(data.x, data.edge_index, return_all_emb=True)

                output2 = discriminator(h2)
                loss_d = criterion(output2.view(-1),
                                   data.x[:, data.sens_idx])
                loss_d.backward()
                optimizer_d.step()

            # train classifier
            classifier.train()
            encoder.eval()
            discriminator.eval()
            for epoch_c in range(0, args.c_epochs):
                optimizer_c.zero_grad()
                h1,h2 = encoder(data.x, data.edge_index,return_all_emb=True)
                output = classifier(h2)
                output2 = discriminator(h2)

                loss_c = F.binary_cross_entropy_with_logits(
                    output[data.train_mask].float(), data.y[data.train_mask].unsqueeze(1).float().to(args.device))\
                         - args.beta*criterion(output2.view(-1),data.x[:, data.sens_idx])

                loss_c.backward()
                optimizer_c.step()

            # train debiaslayer to fool discriminator
            encoder.train()
            discriminator.eval()
            for epoch_g in range(0, args.g_epochs):
                optimizer_e.zero_grad()
                h1,h2 = encoder(data.x, data.edge_index,return_all_emb=True)
                # output1 = discriminator(h2)
                output = classifier(h2)
                output2 = discriminator(h2)
                # 接近于随机预测概率
                loss_de = F.mse_loss(output2.view(-1), 0.5 * torch.ones_like(output2.view(-1))) + args.alpha*F.binary_cross_entropy_with_logits(
                    output[data.train_mask].float(), data.y[data.train_mask].unsqueeze(1).float().to(args.device))

                loss_de.backward()
                optimizer_e.step()

            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_1(classifier,encoder,data, args)
            #
            if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (
                    tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']
                best_val_tradeoff = auc_rocs['val'] + F1s['val'] + \
                                    accs['val'] - (tmp_parity['val'] + tmp_equality['val'])

                print('Epoch: {:04d}'.format(epoch + 1),
                      'acc_test: {:.4f}'.format(test_acc),
                      "roc_test: {:.4f}".format(test_auc_roc),
                      "parity_test: {:.4f}".format(test_parity),
                      "equality_test: {:.4f}".format(test_equality))
                best_debias_weights = encoder.debiaslayer2.weights

                embeddings = h2

        print("****************beta")
        print(args.beta)
        print(args.alpha)

        acc[count] = test_acc
        f1[count] = test_f1
        auc_roc[count] = test_auc_roc
        parity[count] = test_parity
        equality[count] = test_equality
        times[count] = time.time() - start
        # print(best_debias_weights)
        # figure_embedding(best_debias_weights)
        save_dir_d = './debias_weight/'
        judge_dir(save_dir_d)
        np.savetxt(save_dir_d+args.encoder+'_'+args.dataset+'_'+str(count)+'.txt',np.array(best_debias_weights.detach().cpu().numpy()))

        save_dir_1 = save_dir + '/embeddings/'
        judge_dir(save_dir_1)
        np.savetxt(save_dir_1 + args.encoder + '_' + args.dataset + '_' + str(count) + '.txt',
                   np.array(embeddings.detach().cpu().numpy()))

    return acc, f1, auc_roc, parity, equality, times


if __name__=='__main__':
    import warnings
    warnings.filterwarnings('ignore')
    args = parse_args()

    for encoder in ['GCN']:  # 'MLP','GCN','GIN''german','credit'
        args.encoder = encoder
        for dataset in ['pokec_z',]:  # 'bail',,'GCN','GIN''bail','credit'
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
                path =  r"/home/mzx/GNNDebias_1/dataset/pokec/"
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

            acc, f1, auc_roc, parity, equality, times = main_3(pyg_data, args)
            result_save = []
            var_save = []
            print('======' + args.dataset + args.encoder + '======')
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

            save_dir = '/home/mzx/GNNDebias_1/Debias_Results_2023_6_15/Encoder/'+args.encoder
            from utils.utils import judge_dir

            judge_dir(save_dir)
            np.savetxt("{}/{}_{}_res.txt".format(save_dir, args.dataset, 'GNNDebias_1'), np.array(result_save))
            np.savetxt("{}/{}_{}_val.txt".format(save_dir, args.dataset, 'GNNDebias_1'), np.array(var_save))

