#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
from utils.options import args_parser
import numpy as np
from torchvision import datasets, transforms
args=args_parser()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import numpy as np
from torchvision import datasets, transforms
def build_matrix(num_group,num_classes):
    np.random.seed(42)
    num_member=5
    arr=np.arange(1,20)
    distribute=np.random.choice(arr,num_group-1,replace=False)
    distribute=sorted(distribute)

    #distribute=[5,10,15]
    distribute.append(20)
    print(distribute)
    matrix=[[0 for i in range(num_group*num_member+1)]for i in range(num_classes)]
    tag=0
    for k in range(4):
        for i in range(tag,distribute[k]):
            if i == tag:
                label = [np.arange(num_classes) for j in range(20)]
                label[i] = list(set(label[i]) - set([0, 1, 2, 3]))
                temp = set(np.random.choice(label[i], 2, replace=False))
                label[i] = list(set(label[i]) - temp)
                temp = list(temp)

            matrix[k][i ] = random.uniform(0.01, 0.01)
            matrix[temp[0]][i ] = random.uniform(0.01, 0.01)
            matrix[temp[1]][i ] = random.uniform(0.01, 0.01)
        tag = distribute[k]
    for k in range(num_classes):
        temp=sum(matrix[k])
        temp=1-temp
        count=0
        for i in range(len(matrix[k])-1):
            if matrix[k][i]==0:
                count+=1
        for i in range(len(matrix[k])-1):
            if matrix[k][i]==0:
                matrix[k][i]=temp/count

    return np.array(matrix),distribute
def non_iid(train_labels,num_users):
    idxs = np.arange(len(train_labels))
    labels = train_labels.numpy()
    labels_sort=labels.argsort()
    labels_sign=[0 for i in range(10)]
    dict_users=[[]for i in range(num_users)]
    number=1
    count=0
    for i in labels_sort:
        if labels[i]==number:
            labels_sign[number]=count
            number+=1
        count+=1
    print(labels_sign)

def dirichlet_split_noniid(train_labels, n_clients,alpha=1):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    non_iid(train_labels,n_clients)
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少
    '''
    n_clients+=1
    label_distribution=np.array([[0.1,0.3,0,0,0,0,0,0,0,0],
       [0.1,0.3,0,0,0,0,0,0,0,0],
       [0.1,0.3,0.1,0.3,0,0,0,0,0,0],
       [0,0,0.1,0.3,0,0,0,0,0,0],
       [0,0,0.1,0.3,0.1,0.3,0,0,0,0],
       [0,0,0,0,0.1,0.3,0,0,0,0],
       [0,0,0,0,0.1,0.3,0.1,0.3,0,0],
       [0,0,0,0,0,0,0.1,0.3,0.1,0],
       [0,0,0,0,0,0,0.1,0.3,0.1,0],
       [0,0,0,0,0,0,0,0,0.1,0]])
    '''
    '''

    for i in range(n_classes):
        label_distribution[i][i*2]=1.5
        label_distribution[i][i * 2+1]=1.5
        s=sum(label_distribution[i])
        for j in range(len(label_distribution[i])):
            label_distribution[i][j]=label_distribution[i][j]/(s)
    '''

    label_distribution,distribute=build_matrix(4,n_classes)
    n_clients += 1
    print(label_distribution)

    class_idcs = [np.argwhere(train_labels==y).flatten()
           for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(n_clients)]
    client_idcs_test=[[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    for i in range(n_clients):
        temp=set(np.random.choice(client_idcs[i],int(args.test_size*len(client_idcs[i])),replace=False))
        client_idcs[i]=list(set(client_idcs[i])-temp)
        client_idcs_test[i]=list(temp)
    for i in range(n_clients):
        print(len(client_idcs[i]),end=' ')
        print(len(client_idcs_test[i]))
    return client_idcs,client_idcs_test,distribute

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 400, 150
    idx_shard = [i for i in range(num_shards)]
    #dict_users=[[] for _ in range(num_users)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 3, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        dict_users_test = [[] for _ in range(num_users)]
        for i in range(num_users):
            temp = set(np.random.choice(dict_users[i], int(0.2 * len(dict_users[i])), replace=False))
            dict_users[i] = np.array(set(dict_users[i]) - temp)
            dict_users_test[i] = temp
    return dict_users,dict_users_test
import numpy as np
import torch.utils.data as Data
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def iid(dataset, num_users,all_idxs):
    np.random.seed(42)
    num_client=num_users
    # local dataset num
    num_items = int(len(all_idxs) / num_users)
    # dict_users -> 所有客户端数据 samples 的索引字典, all_idxs -> 所有数据 samples 的索引列表
    dict_users = [[] for i in range(num_users)]
    # 基于 clients num 执行循环
    for i in range(num_users):
        # 基于字典 data structure 分配每个 client 的数据 samples -> set() + np.random.choice(ndarray, int)
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # 从所有数据 samples 的索引列表中删除已经分配的数据 samples
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i]=list(dict_users[i])
    dict_users_test = [[] for i in range(num_users)]
    for i in range(num_users):
        temp=set(np.random.choice(dict_users[i],int(args.test_size*len(dict_users[i])),replace=False))
        dict_users[i]=list(set(dict_users[i])-temp)
        dict_users_test[i]=list(temp)
    data_train=[[] for i in range(num_client)]
    data_test=[[] for i in range(num_client)]
    for i in range(num_users):
        X_=[]
        y_=[]
        X_test=[]
        y_test=[]
        for j in range(len(dict_users[i])):
            X_.append(dataset[dict_users[i][j]][0].numpy().tolist())
            y_.append(dataset[dict_users[i][j]][1].item())
        for j in range(len(dict_users_test[i])):
            X_test.append(dataset[dict_users_test[i][j]][0].numpy().tolist())
            y_test.append(dataset[dict_users_test[i][j]][1].item())

        y_=torch.from_numpy(np.array(y_).astype(np.int64))
        X_=torch.from_numpy(np.array(X_).astype(np.float32))

        X_test = torch.from_numpy(np.array(X_test).astype(np.float32))
        y_test = torch.from_numpy(np.array(y_test).astype(np.int64))
        data_train[i]=Data.TensorDataset(X_, y_)
        data_test[i]=Data.TensorDataset(X_test, y_test)
    return data_train,data_test
def non_iid_mpc():#battery,0,ram,13

    spam = pd.read_csv('./double_auction_with_TD3/data/train.csv')  # 训练集
    X = spam.iloc[:, 0:20].values  # 取前26列作为X
    y = spam['price_range'].values  # 训练集标签
    loc1 = spam['ram'].values
    loc2 = spam['blue'].values
    loc3 = spam['battery_power'].values
    scales = MinMaxScaler(feature_range=(0, 1))
    X = scales.fit_transform(X)
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.int64))
    dataset = Data.TensorDataset(X, y)
    arr_1=loc3.argsort()
    arr_2=loc2.argsort()
    arr_3=loc3.argsort()
    temp_1,temp_2,temp_3,temp_4=[],[],[],[]
    idx_1,idx_2,idx_3,idx_4,idx_5,idx_6,idx_7,idx_8=[],[],[],[],[],[],[],[]
    for i in range(int(len(dataset)/4)):
        temp_1.append(arr_1[i])
    for i in range(int(len(dataset)/4),2*int(len(dataset)/4)):
        temp_2.append(arr_1[i])
    for i in range(2*int(len(dataset)/4),3*int(len(dataset)/4)):
        temp_3.append(arr_1[i])
    for i in range(3*int(len(dataset)/4),4*int(len(dataset)/4)):
        temp_4.append(arr_1[i])
    for i in temp_1:
        if loc2[i]==0:
            idx_1.append(i)
        else:
            idx_2.append(i)
    for i in temp_2:
        if loc2[i]==0:#>=loc2[arr_2[int(len(arr_2)/2)]]:
            idx_3.append(i)
        else:
            idx_4.append(i)
    for i in temp_3:
        if loc2[i]==0:
            idx_5.append(i)
        else:
            idx_6.append(i)
    for i in temp_4:
        if loc2[i]==0:#>=loc2[arr_2[int(len(arr_2)/2)]]:
            idx_7.append(i)
        else:
            idx_8.append(i)
    train,test=[],[]
    print(idx_1)
    print(idx_2)
    print(idx_3)
    print(idx_4)
    print(idx_5)
    print(idx_6)
    print(idx_7)
    print(idx_8)

    train_,test_=iid(dataset,2,idx_1)
    for i in train_:
        train.append(i)
    for i in test_:
        test.append(i)
    train_, test_ = iid(dataset, 2, idx_2)
    for i in train_:
        train.append(i)
    for i in test_:
        test.append(i)
    train_, test_ = iid(dataset, 2, idx_3)
    for i in train_:
        train.append(i)
    for i in test_:
        test.append(i)
    train_, test_ = iid(dataset, 3, idx_4)
    for i in train_:
        train.append(i)
    for i in test_:
        test.append(i)
    train_, test_ = iid(dataset, 3, idx_5)
    for i in train_:
        train.append(i)
    for i in test_:
        test.append(i)
    train_, test_ = iid(dataset, 3, idx_6)
    for i in train_:
        train.append(i)
    for i in test_:
        test.append(i)
    train_, test_ = iid(dataset, 3, idx_7)
    for i in train_:
        train.append(i)
    for i in test_:
        test.append(i)
    train_, test_ = iid(dataset, 2, idx_8)
    for i in train_:
        train.append(i)
    for i in test_:
        test.append(i)
    return train,test


