#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 10:13
# @Author  : zhixiuma
# @File    : load_data.py
# @Project : Test
# @Software: PyCharm

import os
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
# from torch_geometric.data.dataset import
from torch_geometric.utils import subgraph

GROUP=False


def build_relationship(x,thresh=0.25):
    df_euclid = pd.DataFrame(1/(1+distance_matrix(x.T.T,x.T.T)),columns=x.T.columns,index = x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind,:])[-2]
        neig_id = np.where(df_euclid[ind,:] > thresh*max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)

        for neig in neig_id:
            if neig!=ind:
                idx_map.append([ind,neig])
    idx_map = np.array(idx_map)

    return idx_map

def load_credit(dataset,sens_attr='Age',predict_attr='NoDefaultNextMonth',path=r'/home/mzx/GNNDebias_1/dataset\dataset\dataset\credit',label_number=100):
    idx_features_labels = pd.read_csv(os.path.join(path,'{}.csv'.format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unorderd = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')

    features = sp.csr_matrix(idx_features_labels[header],dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j:i for i,j in enumerate(idx)}
    # print(idx_map)
    edges = np.array(list(map(idx_map.get,edges_unorderd.flatten())),dtype=int).reshape(edges_unorderd.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]),(edges[:,0],edges[:,1])),shape=(labels.shape[0],labels.shape[0]),dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    print(len(labels))
    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5*len(label_idx_0)),label_number//2)],label_idx_1[:min(int(0.5*len(label_idx_1)),label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5*len(label_idx_0)):int(0.75*len(label_idx_0))],label_idx_1[int(0.5*len(label_idx_1)):int(0.75*len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75*len(label_idx_0)):],label_idx_1[int(0.75*len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if GROUP:
        sens_idx_0_list = torch.nonzero(sens)
        mask = sens==0
        sens_idx_1_list = torch.nonzero(mask)
        sub_edges_0,_=subgraph(subset=sens_idx_0_list,edge_index=torch.LongTensor(adj.nonzero()))
        sub_edges_1, _ = subgraph(subset=sens_idx_1_list, edge_index=torch.LongTensor(adj.nonzero()))
        # print(sens_idx_0_list)
        print('sens_0_node_ratio:',len(sens_idx_0_list)/len(idx))
        print('sens_1_node_ratio:',len(sens_idx_1_list)/len(idx))
        print('sens_0_intra_group_edges_ratio:', len(sub_edges_0[0]) + len(sub_edges_1[0]))
        print('sens_1_inter_group_edges_ratio:',
              len(adj.nonzero()[0]) - len(sub_edges_1[0]) - len(sub_edges_0[0]))

        print('sens_0_intra_group_edges_ratio:', (len(sub_edges_0[0]) + len(sub_edges_1[0])) / len(adj.nonzero()[0]))
        print('sens_1_inter_group_edges_ratio:',
              (len(adj.nonzero()[0]) - len(sub_edges_1[0]) - len(sub_edges_0[0])) / len(adj.nonzero()[0]))
        # print(len(sens_idx_1_list))

        # np.savetxt()


        # torch.nonzero(a==10).squeeze()
    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="dataset/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    print(len(labels))
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    print(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if GROUP:
        sens_idx_0_list = torch.nonzero(sens)
        mask = sens == 0
        sens_idx_1_list = torch.nonzero(mask)
        sub_edges_0, _ = subgraph(subset=sens_idx_0_list, edge_index=torch.LongTensor(adj.nonzero()))
        sub_edges_1, _ = subgraph(subset=sens_idx_1_list, edge_index=torch.LongTensor(adj.nonzero()))
        # print(sens_idx_0_list)
        print('sens_0_node_ratio:', len(sens_idx_0_list) / len(idx))
        print('sens_1_node_ratio:', len(sens_idx_1_list) / len(idx))
        print('sens_0_intra_group_edges_ratio:', len(sub_edges_0[0]) + len(sub_edges_1[0]))
        print('sens_1_inter_group_edges_ratio:',
              len(adj.nonzero()[0]) - len(sub_edges_1[0]) - len(sub_edges_0[0]))

        print('sens_0_intra_group_edges_ratio:', (len(sub_edges_0[0]) + len(sub_edges_1[0])) / len(adj.nonzero()[0]))
        print('sens_1_inter_group_edges_ratio:',
              (len(adj.nonzero()[0]) - len(sub_edges_1[0]) - len(sub_edges_0[0])) / len(adj.nonzero()[0]))

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="dataset/german/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # sensitive feature removal
    # header.remove('Gender')

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0


    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    # print(labels)
    labels = labels.astype(int)
    print(np.unique(labels))
    labels[labels == -1] = 0
    print(np.unique(labels))
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if GROUP:
        sens_idx_0_list = torch.nonzero(sens)
        mask = sens == 0
        sens_idx_1_list = torch.nonzero(mask)
        sub_edges_0, _ = subgraph(subset=sens_idx_0_list, edge_index=torch.LongTensor(adj.nonzero()))
        sub_edges_1, _ = subgraph(subset=sens_idx_1_list, edge_index=torch.LongTensor(adj.nonzero()))
        # print(sens_idx_0_list)
        print('sens_0_node_ratio:', len(sens_idx_0_list) / len(idx))
        print('sens_1_node_ratio:', len(sens_idx_1_list) / len(idx))
        print('sens_0_intra_group_edges_ratio:', len(sub_edges_0[0]) + len(sub_edges_1[0]))
        print('sens_1_inter_group_edges_ratio:',
              len(adj.nonzero()[0]) - len(sub_edges_1[0]) - len(sub_edges_0[0]))

        print('sens_0_intra_group_edges_ratio:', (len(sub_edges_0[0]) + len(sub_edges_1[0])) / len(adj.nonzero()[0]))
        print('sens_1_inter_group_edges_ratio:',
              (len(adj.nonzero()[0]) - len(sub_edges_1[0]) - len(sub_edges_0[0])) / len(adj.nonzero()[0]))

    return adj, features, labels, idx_train, idx_val, idx_test, sens

def make_dataset_1M(load_sidechannel=False):
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', names=r_cols,
			  encoding='latin-1')
    shuffled_ratings = ratings.sample(frac=1).reset_index(drop=True)
    train_cutoff_row = int(np.round(len(shuffled_ratings)*0.9))
    train_ratings = shuffled_ratings[:train_cutoff_row]
    test_ratings = shuffled_ratings[train_cutoff_row:]
    if load_sidechannel:
        u_cols = ['user_id','sex','age','occupation','zip_code']
        m_cols = ['movie_id','title','genre']
        users = pd.read_csv('./ml-1m/users.dat', sep='::', names=u_cols,
                            encoding='latin-1', parse_dates=True)
        movies = pd.read_csv('./ml-1m/movies.dat', sep='::', names=m_cols,
                            encoding='latin-1', parse_dates=True)

    train_ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
    train_ratings_matrix = train_ratings.pivot_table(index=['movie_id'],\
            columns=['user_id'],values='rating').reset_index(drop=True)
    test_ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
    columnsTitles=["user_id","rating","movie_id"]
    train_ratings=train_ratings.reindex(columns=columnsTitles)-1
    test_ratings=test_ratings.reindex(columns=columnsTitles)-1
    users.user_id = users.user_id.astype(np.int64)
    movies.movie_id = movies.movie_id.astype(np.int64)
    users['user_id'] = users['user_id'] - 1
    movies['movie_id'] = movies['movie_id'] - 1

    if load_sidechannel:
        return train_ratings,test_ratings,users,movies
    else:
        return train_ratings,test_ratings

SENSITIVE_ATTR_DICT = {
    'movielens': ['gender', 'occupation', 'age'],
    'pokec': ['gender', 'region', 'AGE'],
    'pokec-z': ['gender', 'region', 'AGE'],
    'pokec-n': ['gender', 'region', 'AGE']
}
#
def load_nba(dataset, sens_attr='country', predict_attr='SALRAY', path="../dataset/NBA/", label_number=100, sens_number=50,
               test_idx=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    # print(idx_features_labels)
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
    # header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    # print(idx_features_labels['MP'])
    labels = idx_features_labels[predict_attr].values

    # build graph
    node_ids = list(idx_features_labels['user_id'])
    new_ids = list(range(len(node_ids)))

    id_map = dict(zip(node_ids, new_ids))
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=np.int64)

    edges = np.array(list(map(id_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(20)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)), label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(20)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # random.shuffle(sens_idx)
    if GROUP:
        sens_idx_0_list = torch.nonzero(sens)
        mask = sens == 0
        sens_idx_1_list = torch.nonzero(mask)
        sub_edges_0, _ = subgraph(subset=sens_idx_0_list, edge_index=torch.LongTensor(adj.nonzero()))
        sub_edges_1, _ = subgraph(subset=sens_idx_1_list, edge_index=torch.LongTensor(adj.nonzero()))
        # print(sens_idx_0_list)
        print('sens_0_node_ratio:', len(sens_idx_0_list) / len(node_ids))
        print('sens_1_node_ratio:', len(sens_idx_1_list) / len(node_ids))
        print('sens_0_intra_group_edges_ratio:', len(sub_edges_0[0]) + len(sub_edges_1[0]))
        print('sens_1_inter_group_edges_ratio:',
              len(adj.nonzero()[0]) - len(sub_edges_1[0]) - len(sub_edges_0[0]))

        print('sens_0_intra_group_edges_ratio:', (len(sub_edges_0[0]) + len(sub_edges_1[0])) / len(adj.nonzero()[0]))
        print('sens_1_inter_group_edges_ratio:',
              (len(adj.nonzero()[0]) - len(sub_edges_1[0]) - len(sub_edges_0[0])) / len(adj.nonzero()[0]))

    return adj, features, labels, idx_train, idx_val, idx_test, sens



def load_pokec(dataset, sens_attr="region", predict_attr="I_am_working_in_field", path="dataset/pokec-z/", label_number=500):
    # print('Converting to csv format...')
    # print(dataset)
    # if dataset == 'pokec_z':
    #     dataset = 'region_job'
    # else:
    #     dataset = 'region_job_2'

    if dataset =='pokec_z':
        edge_file = "{}/region_job_relationship.txt".format(path)
        node_file = '{}/region_job.csv'.format(path)
    elif dataset =='pokec_n':
        edge_file = "{}/region_job_2_relationship.txt".format(path)
        node_file = '{}/region_job_2.csv'.format(path)

    edges_unordered= np.genfromtxt(edge_file).astype(int)

    idx_features_labels = pd.read_csv(node_file,sep=',',header=0,engine='python')
    print('---raw data loaded')

    node_ids = list(idx_features_labels['user_id'])
    new_ids = list(range(len(node_ids)))

    id_map = dict(zip(node_ids, new_ids))
    feature_ls = ['user_id',]# sens_attr,predict_attr

    header = list(idx_features_labels.columns)
    for f in feature_ls:
        print(f)
        header.remove(f)
    # header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    # print(features.shape)# (67796, 273)
    labels = idx_features_labels[predict_attr].values
    labels = labels.astype(int)

    # labels[labels >= 1] = 1
    labels[labels > 1] = 1
    # if sens_attr:
    #     sens[sens > 0] = 1
    edges = np.array(list(map(id_map.get, edges_unordered.flatten())),
                     ).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    print('sens:',np.unique(sens))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # if gra
    if GROUP:
        sens_idx_0_list = torch.nonzero(sens)
        mask = sens == 0
        sens_idx_1_list = torch.nonzero(mask)
        sub_edges_0, _ = subgraph(subset=sens_idx_0_list, edge_index=torch.LongTensor(adj.nonzero()))
        sub_edges_1, _ = subgraph(subset=sens_idx_1_list, edge_index=torch.LongTensor(adj.nonzero()))
        # print(sens_idx_0_list)
        print('sens_0_node_ratio:', len(sens_idx_0_list) / len(node_ids))
        print('sens_1_node_ratio:', len(sens_idx_1_list) / len(node_ids))
        print('sens_0_intra_group_edges_ratio:', len(sub_edges_0[0]) + len(sub_edges_1[0]))
        print('sens_1_inter_group_edges_ratio:',
              len(adj.nonzero()[0]) - len(sub_edges_1[0]) - len(sub_edges_0[0]))

        print('sens_0_intra_group_edges_ratio:', (len(sub_edges_0[0]) + len(sub_edges_1[0])) / len(adj.nonzero()[0]))
        print('sens_1_inter_group_edges_ratio:',
              (len(adj.nonzero()[0]) - len(sub_edges_1[0]) - len(sub_edges_0[0])) / len(adj.nonzero()[0]))

    # print(labels.shape)
    return adj, features, labels, idx_train, idx_val, idx_test, sens


class Dataset():
    def __init__(self,name='dataset',sens_attr="WHITE" ,predict_attr="RECID", path="D:\\\TASKS\\\graph_fair\\Test\\dataset\\bail\\bail\\", label_number=1000):
        self.name= name
        # print(self.name)
        self.sens_attr = sens_attr

        self.predict_attr = predict_attr
        self.path = path
        self.label_number = label_number

        self.load_data()

    def load_data(self):
        if self.name=='bail':
            self.adj,self.features,self.labels,self.idx_train,self.idx_val,self.idx_test,self.sens = load_bail(
                dataset=self.name,
                   sens_attr=self.sens_attr,
                   predict_attr=self.predict_attr,
                   path=self.path,
                   label_number=self.label_number)
            self.sens_idx = 0
            # self.features = self.feature_norm(self.features)
            # self.features[:, self.sens_idx] = self.features[:, self.sens_idx]
            self.norm_features = self.feature_norm(self.features)
            self.norm_features[:, self.sens_idx] = self.features[:, self.sens_idx]
            self.features = self.norm_features
            print(np.unique(self.labels))
            self.counter_feature = self.flip_feature(self.features,self.sens_idx)
            self.pyg_data = self.toPygData(adj=self.adj,features=self.features,labels=self.labels,
                           idx_train=self.idx_train,idx_val=self.idx_val,idx_test=self.idx_val,
                           sens_idx=self.sens_idx,counter_feature=self.counter_feature)

        elif self.name=='credit':
            self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test, self.sens = load_credit(
                dataset=self.name,
                sens_attr=self.sens_attr,
                predict_attr=self.predict_attr,
                path=self.path,
                label_number=self.label_number
            )
            self.sens_idx = 1
            # print(self.features[:, self.sens_idx])
            self.norm_features = self.feature_norm(self.features)
            self.norm_features[:, self.sens_idx] = self.features[:, self.sens_idx]
            # print(np.unique(self.labels))
            self.features = self.norm_features
            self.counter_feature = self.flip_feature(self.features, self.sens_idx)
            # print(self.features[:,self.sens_idx])
            # print(self.counter_feature[:,self.sens_idx])
            print(np.unique(self.labels))
            self.pyg_data = self.toPygData(adj=self.adj, features=self.features, labels=self.labels,
                                           idx_train=self.idx_train, idx_val=self.idx_val, idx_test=self.idx_val,
                                           sens_idx=self.sens_idx, counter_feature=self.counter_feature)

        elif self.name=='german':
            self.sens_idx = 0
            self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test, self.sens = load_german(
                dataset=self.name,
                sens_attr=self.sens_attr,
                predict_attr=self.predict_attr,
                path=self.path,
                label_number=self.label_number
            )
            # print(self.labels)
            # print(np.unique(self.labels))
            self.counter_feature = self.flip_feature(self.features, self.sens_idx)
            self.pyg_data = self.toPygData(adj=self.adj, features=self.features, labels=self.labels,
                                           idx_train=self.idx_train, idx_val=self.idx_val, idx_test=self.idx_val,
                                           sens_idx=self.sens_idx, counter_feature=self.counter_feature)

        elif self.name=='pokec_n' or self.name=='pokec_z':
            # print()
            self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test, self.sens = load_pokec(
                dataset=self.name,
                sens_attr=self.sens_attr,
                predict_attr=self.predict_attr,
                path=self.path,
                label_number=self.label_number
            )
            print(np.unique(self.labels))
            self.sens_idx = 4
            # self.labels_1 = one_hot(torch.Tensor(self.labels),dtype=torch.long)
            self.counter_feature = self.flip_feature(self.features, self.sens_idx)
            self.pyg_data = self.toPygData(adj=self.adj, features=self.features, labels=self.labels,
                                           idx_train=self.idx_train, idx_val=self.idx_val, idx_test=self.idx_val,
                                           sens_idx=self.sens_idx, counter_feature=self.counter_feature)

        elif self.name=='nba':
            self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test, self.sens = load_nba(
                dataset=self.name,
                sens_attr=self.sens_attr,
                predict_attr=self.predict_attr,
                path=self.path,
                label_number=self.label_number
            )
            self.sens_idx = 1
            # print(np.unique(self.labels))
            self.counter_feature = self.flip_feature(self.features, self.sens_idx)
            self.pyg_data = self.toPygData(adj=self.adj, features=self.features, labels=self.labels,
                                           idx_train=self.idx_train, idx_val=self.idx_val, idx_test=self.idx_val,
                                           sens_idx=self.sens_idx, counter_feature=self.counter_feature)
            print(self.adj.shape)

    def feature_norm(self,features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1

    # 反事实
    def flip_feature(self,x, sens_idx, sens_flag=True):
        x = x.clone()
        if sens_flag:
            x[:, sens_idx] = 1 - x[:, sens_idx]
        return x

    # def to(self,device):
    def index_to_mask(self,index, size):
        mask = torch.zeros((size,), dtype=torch.bool)
        mask[index] = 1
        return mask

    def toPygData(self,adj, features, labels, idx_train, idx_val, idx_test,sens_idx,counter_feature):
        # Dpr2Pyg
        edge_index = torch.LongTensor(adj.nonzero())
        # by default, the features in pyg data is dense
        if sp.issparse(features):
            x = torch.FloatTensor(features.todense()).float()
        else:
            x = torch.FloatTensor(features).float()
        # print(labels)
        y = torch.LongTensor(labels)

        data = Data(x=x, edge_index=edge_index, y=y,sens_idx=sens_idx,counter_feature=counter_feature)
        train_mask = self.index_to_mask(idx_train, size=y.size(0))
        val_mask = self.index_to_mask(idx_val, size=y.size(0))
        test_mask = self.index_to_mask(idx_test, size=y.size(0))
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.sens = self.sens
        data.sens_idx = sens_idx
        return data

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2},labels={3},idx_train={4},idx_val={5},idx_test={6},sens_idx={7},sens={8},predict_attr={9}))'.format(
            'Dataset_:'+self.name, self.adj.shape, self.features.shape,
            self.labels.shape,self.idx_train.shape,self.idx_val.shape,self.idx_test.shape,self.sens_idx,self.sens_attr,self.predict_attr)


def deal_with_sens():
    load_credit(dataset='Credit', sens_attr='Age', predict_attr='NoDefaultNextMonth', path=r'/dataset/credit', label_number=6000)
    load_german(dataset='german', sens_attr="Gender", predict_attr="GoodCustomer", path=r'/dataset/german', label_number=100)
    load_bail(dataset='bail', sens_attr="WHITE", predict_attr="RECID", path=r'/dataset/bail', label_number=100)
    # load_nba(dataset='nba', sens_attr='country', predict_attr='SALARY', path=r'/dataset/NBA', label_number=100,
    #          sens_number=50,
    #          test_idx=False)
    load_pokec(dataset='pokec_z', sens_attr="region", predict_attr="I_am_working_in_field", path=r'/dataset/pokec',
               label_number=500)

    load_pokec(dataset='pokec_n', sens_attr="region", predict_attr="I_am_working_in_field", path=r'/dataset/pokec',
               label_number=500)

def read_data():
    from trainer_args import parse_args
    args = parse_args()
    #
    args.dataset = 'credit'
    print('**********************' + args.dataset + '**********************')
    data = Dataset(name=args.dataset, sens_attr='Age', predict_attr='NoDefaultNextMonth',
                   path=r"/home/mzx/GNNDebias_1/dataset/credit", label_number=6000)
    print(data)
    print(data.features.shape[0])
    print(data.features.shape[1])
    print(len(data.adj.nonzero()[0]))
    print(data.sens_attr)
    print(np.unique(data.labels))


    args.dataset = 'german'
    print('**********************' + args.dataset + '**********************')
    data = Dataset(name=args.dataset, sens_attr="Gender", predict_attr="GoodCustomer", path=r"/home/mzx/GNNDebias_1/dataset/german", label_number=100)
    print(data)
    print(data.features.shape[0])
    print(data.features.shape[1])
    print(len(data.adj.nonzero()[0]))
    print(data.sens_attr)
    print(np.unique(data.labels))
    #
    args.dataset = 'bail'
    print('**********************' + args.dataset + '**********************')
    data = Dataset(name=args.dataset, sens_attr="WHITE", predict_attr="RECID", path=r"/home/mzx/GNNDebias_1/dataset/bail", label_number=100)
    print(data)
    print(data.features.shape[0])
    print(data.features.shape[1])
    print(len(data.adj.nonzero()[0]))
    print(data.sens_attr)
    print(np.unique(data.labels))
    # #
    args.dataset = 'pokec_z'
    print('**********************' + args.dataset + '**********************')
    data = Dataset(name=args.dataset, sens_attr="region", predict_attr="I_am_working_in_field",
                   path=r"/home/mzx/GNNDebias_1/dataset/pokec", label_number=500)
    print(data)
    print(data.features.shape[0])
    print(data.features.shape[1])
    print(len(data.adj.nonzero()[0]))
    print(data.sens_attr)
    print(np.unique(data.labels))
    #   NumNodes: 67796
    #   NumEdges: 1303712
    #   NumFeats: 273

    args.dataset = 'pokec_n'
    print('**********************' + args.dataset + '**********************')
    data = Dataset(name=args.dataset, sens_attr="region", predict_attr="I_am_working_in_field",
                   path=r"/home/mzx/GNNDebias_1/dataset/pokec", label_number=500)
    print(data)
    print(data.features.shape[0])
    print(data.features.shape[1])
    print(len(data.adj.nonzero()[0]))
    print(data.sens_attr)
    print(np.unique(data.labels))

    # args.dataset = 'nba'
    # print('**********************' + args.dataset + '**********************')
    # data = Dataset(name=args.dataset, sens_attr="country", predict_attr="SALARY",
    #                path=r"/home/mzx/GNNDebias_1/dataset\dataset\nba", label_number=100)
    # print(data)
    # print(data.features.shape[0])
    # print(data.features.shape[1])
    # print(len(data.adj.nonzero()[0]))
    # print(data.sens_attr)
    # print(np.unique(data.labels))

if __name__=='__main__':
   import warnings
   warnings.filterwarnings('ignore')
   # deal_with_sens()
   read_data()