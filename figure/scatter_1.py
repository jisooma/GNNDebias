#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/13 15:03
# @Author  : zhixiuma
# @File    : scatter_1.py
# @Project : Test
# @Software: PyCharm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.utils import judge_dir

colormap = list(plt.get_cmap('tab10')(np.linspace(0, 1, 10)))
marker =['s','p','*','H','P','1','2','3','4']

save_dir = './trade_off_1/'
judge_dir(save_dir)
def scatter_figure(x,y,datset,base_model):
    model_list = ['Vanilla','FairGNN', 'NIFTY', 'FairVGNN', 'GNNDebias']
    plt.figure(figsize=(7, 5))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=colormap[i], edgecolors=colormap[i],s=600,alpha=0.5,marker=marker[i])
    plt.legend(model_list, loc=1, borderaxespad=0,ncol=5,columnspacing=0.2,bbox_to_anchor=(0.95, 1.08),framealpha=0.1)
    plt.ylabel('ΔSP+ΔEO(←)',fontsize=15)
    plt.xlabel('ACC+AUC+F1(→)',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('{}/{}_{}.png'.format(save_dir, dataset,base_model))
    plt.show()
    #

if __name__=='__main__':
    # from GNNDebias_1_Unlearning.utils.utils import judge_dir
    #
    # path_acc_dir_1 = r"/home/mzx/GNNDebias_1Original_Results_2023_6_13"
    # path_acc_dir = r"/home/mzx/GNNDebias_1Debias_Results_2023_6_13\Encoder"
    # save_path = path_acc_dir + "\debias_tables/"
    # judge_dir(save_path)
    # debias_model_list = ['FairGNN', 'NIFTY', 'FairVGNN', 'GNNDebias_1', ]  # 'Vanilla','GNNDebias_1','FairVGNN','credit'
    # dataset_list = ['pokec_z', 'pokec_n']  # 'credit','bail','german'
    # base_model_list = ['GCN',]
    # for base_model in base_model_list:
    #     for dataset in dataset_list:
    #         acc_var_all = []
    #         performance = []
    #         fair = []
    #         acc_list_1 = np.loadtxt('{}/{}/{}_{}_res.txt'.format(path_acc_dir_1, base_model, dataset,
    #                                                              'GNNDebias_1'))
    #
    #         performance.append(acc_list_1[1])
    #         fair.append(acc_list_1[3])
    #
    #         for debias_model in debias_model_list:
    #             acc_list = np.loadtxt('{}/{}/{}_{}_res.txt'.format(path_acc_dir, base_model, dataset,
    #                                                                  debias_model))
    #             performance.append(acc_list[1])
    #             fair.append(acc_list[3])
    #
    #         scatter_figure(performance,fair,dataset,base_model)

    from utils.utils import judge_dir

    path_acc_dir_1 = r"/home/mzx/GNNDebias_1/Original_Results_2023_6_15"
    path_acc_dir = r"/home/mzx/GNNDebias_1/Debias_Results_2023_6_15/Encoder"
    save_path = path_acc_dir + "/ebias_tables/"
    judge_dir(save_path)
    debias_model_list = ['FairGNN', 'NIFTY', 'FairVGNN', 'GNNDebias_1', ]  # 'Vanilla','GNNDebias_1','FairVGNN','credit'
    dataset_list = ['pokec_z',]  #'bail','credit','pokec_n'
    base_model_list = ['GCN',]# 'GCN''GIN','SAGE',
    for base_model in base_model_list:
        for dataset in dataset_list:
            acc_var_all = []
            performance = []
            fair = []
            acc_list_1 = np.loadtxt('{}/{}/{}_{}_res.txt'.format(path_acc_dir_1, base_model, dataset,
                                                                 'GNNDebias_1'))

            performance.append(acc_list_1[0]+acc_list_1[1]+acc_list_1[2])
            fair.append(acc_list_1[3]+acc_list_1[4])

            for debias_model in debias_model_list:
                acc_list = np.loadtxt('{}/{}/{}_{}_res.txt'.format(path_acc_dir, base_model, dataset,
                                                                     debias_model))
                performance.append(acc_list[0]+acc_list[1]+acc_list[2])
                fair.append(acc_list[3]+acc_list[4])

            scatter_figure(performance,fair,dataset,base_model)
