#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 09:53
# @Author  : zhixiuma
# @File    : figure.py
# @Project : Test
# @Software: PyCharm
# import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import judge_dir

def figure_embedding(embeddings,count):
    # save_dir = './figure/output_log_520/'+attack+'/'+defense+'/'+dataset
    # judge_dir(save_dir)
    print(embeddings.shape)
    ax = sns.heatmap(embeddings
                     ,cmap="YlGnBu",vmin=-2.5,vmax=2.5)
    # plt.title('Truth Label is '+str(label))
    plt.xlabel('Hidden Channels')
    plt.ylabel('Hidden Channels')
    # plt.yticks(rotation=0)
    plt.savefig('{}/{}.png'.format(save_dir,str(count)))
    plt.show()



if __name__=='__main__':
    save_dir = '../debias_weight/'
    args = {}
    judge_dir(save_dir)
    for encoder in ['GCN']:  # 'MLP','GCN','GIN''german','credit'
        # args.encoder = encoder
        for dataset in ['pokec_z',]:  # 'bail',,'GCN','GIN''pokec_n'
            w_list = []
            for count in [0,1,2,3,4]:
                # args.dataset = dataset
                list1 = []
                w = np.loadtxt(save_dir + encoder + '_' + dataset + '_' + str(count) + '.txt')
                list1.append(w.sum(0))
                print(w.sum(0).shape)
                figure_embedding(np.array(list1),count)