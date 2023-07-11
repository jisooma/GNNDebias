# _*_codeing=utf-8_*_
# @Time:2022/9/30  09:41
# @Author:mazhixiu
# @File:TSNE.py
import matplotlib.pyplot as plt
import os

import numpy as np
from sklearn.manifold import TSNE
"""
节点降维分类图:
TSNE:TSNE将数据点之间的相似度转化位条件概率，原始空间中数据点的相似度由高斯联合分布表示，嵌入空间中数据点的相似度由t分布表示；
通过原始空间和嵌入空间的联合概率分布的KL散度来评估嵌入效果的好坏，即将KL散度的函数作为损失函数，通过梯度下降算法最小化损失函数，最终获得收敛结果。
n_components:指定想要降的维度。
perlexity：如何在局部或全局位面上平衡关注点，也就是对每个点周围邻居数量猜测
fit_transform(X[,y])方法将X嵌入低维空间并返回转换后的输出。
"""
import matplotlib
from utils.utils import judge_dir
matplotlib.rc("font", family='Microsoft YaHei')
# colormap = list(plt.get_cmap('tab10')(np.linspace(0, 1, 10)))
# colormap = ['','ivory']
colormap = np.random.rand(3)
# marker=['s','+','*']
# cmap = plt.cm.get_cmap('cividis_r')
# def tsne_node_features(labels,output,d_name,model=None,count=None):
#
#     experiment_output_dir = './TSNE_figure/'
#     judge_dir(experiment_output_dir)
#     # print(labels.shape)
#     num_labels_type = len(np.unique(labels))
#     z = TSNE(n_components=2).fit_transform(output)
#     # print(z.shape)
#     # plt.figure(figsize=(10,10))
#     # plt.xticks([])
#     # plt.yticks([])
#
#     color_labels_0 = []
#     color_labels_1 = []
#     color_labels_2 = []
#     idx_l0 = []
#     idx_l1 = []
#     idx_l2 = []
#     for i,l in enumerate(labels):
#         if l==0:
#             idx_l0.append(i)
#             color_labels_0.append(colormap[0])
#         if l==1:
#             idx_l1.append(i)
#             color_labels_1.append(colormap[1])
#         if l==-1:
#             idx_l2.append(i)
#             color_labels_2.append(colormap[2])
#
#     # for i,l in enumerate(labels):
#     # print(z[idx_l0].shape)
#     z_0 = np.array(z[idx_l0])
#     z_1 = np.array(z[idx_l1])
#     z_2 = np.array(z[idx_l2])
#     # print(marker[0])
#     # print(color_labels_0[np.array(idx_l0)])
#     plt.scatter(z_0[:,0],z_0[:,1],c=color_labels_0, cmap='Set3')
#     plt.scatter(z_1[:, 0], z_1[:, 1], c=color_labels_1, cmap='Set3')
#     if num_labels_type==3:
#         plt.scatter(z_2[:, 0], z_2[:, 1], c=color_labels_2, cmap='Set3')
#     # plt.title(model+'在' + attack+'下的'+'数据集'+d_name+'的TSNE降维',fontsize= 20)
#     # plt.legend()
#
#     plt.savefig(os.path.join(experiment_output_dir, d_name+'_'+model+"_"+str(count)) + '.jpg',
#                     dpi=500, bbox_inches='tight')
#     plt.show()

def tsne_node_features(labels,output,d_name,model=None,count=None):

    experiment_output_dir = './TSNE_figure/'
    judge_dir(experiment_output_dir)
    # print(labels.shape)
    num_labels_type = len(np.unique(labels))
    z = TSNE(n_components=2,perplexity=40).fit_transform(output)

    color_labels = []
    t0=0
    t1=0
    t2=0
    for i,l in enumerate(labels):
        if l==0:
            t0=t0+1
            color_labels.append(colormap[0])
        if l==1:
            t1=t1+1
            color_labels.append(colormap[1])
        if l==-1:
            t2=t2+1
            color_labels.append(colormap[2])
    print(t0)
    print(t1)
    print(t2)
    # print(marker[0])
    # print(color_labels_0[np.array(idx_l0)])
    plt.scatter(z[:,0],z[:,1],c=color_labels, cmap='Set3',alpha=0.5,s=5)
    # plt.scatter(z_1[:, 0], z_1[:, 1], c=color_labels_1, cmap='Set3')
    # if num_labels_type==3:
    #     plt.scatter(z_2[:, 0], z_2[:, 1], c=color_labels_2, cmap='Set3')
    # plt.title(model+'在' + attack+'下的'+'数据集'+d_name+'的TSNE降维',fontsize= 20)
    # plt.legend()

    plt.savefig(os.path.join(experiment_output_dir, d_name+'_'+model+"_"+str(count)) + '.jpg',
                    dpi=500, bbox_inches='tight')
    plt.show()


if __name__=='__main__':
    import warnings
    warnings.filterwarnings('ignore')

    from utils.utils import judge_dir
    from load_dataset.load_data import Dataset

    path_acc_dir_1 = r"/home/mzx/GNNDebias_1/Original_Results_2023_6_15/GCN/embeddings"
    path_acc_dir = r"/home/mzx/GNNDebias_1/Debias_Results_2023_6_15/GCN/embeddings"
    dataset_list = ['pokec_n','pokec_z','credit','bail']#
    for dataset in dataset_list:
        dataset_1 = dataset

        if dataset == 'german':
            sens_attr = "Gender"
            predict_attr = "GoodCustomer"
            path = r"/home/mzx/GNNDebias_1/dataset/german/"
            label_number = 1000
        elif dataset == 'bail':
            sens_attr = "WHITE"
            predict_attr = "RECID"
            path = r"/home/mzx/GNNDebias_1/dataset/bail/"
            label_number = 1000
        elif dataset == 'credit':
            sens_attr = 'Age'
            predict_attr = 'NoDefaultNextMonth'
            path = r"/home/mzx/GNNDebias_1/dataset/credit/"
            label_number = 100
        elif dataset != 'nba':
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 500
            seed = 20
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

        print('**********************' + dataset + '**********************')
        data = Dataset(name=dataset, sens_attr=sens_attr, predict_attr=predict_attr,
                       path=path, label_number=label_number)
        print(data)
        for i in [0,1,2,3,4]:
            dataset = dataset_1
            embeddings_1 = np.loadtxt('{}//{}_{}_{}.txt'.format(path_acc_dir_1,'GCN', dataset,
                                                                   str(i)))
            embeddings_2 = np.loadtxt('{}//{}_{}_{}.txt'.format(path_acc_dir, 'GCN', dataset,
                                                                 str(i)))
            #
            print(embeddings_1.shape)
            print(data.sens_idx)
            print(np.unique(data.features[:,data.sens_idx]))
            tsne_node_features(labels=data.sens,output=embeddings_1,d_name=dataset,model='GCN',count=i)
            tsne_node_features(labels=data.sens, output=embeddings_2, d_name=dataset, model='GNNDebias_1', count=i)
