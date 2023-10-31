#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from utils.sampling import *
from FL.Update import LocalUpdate
from FL.Nets import *
from FL.Fed import *
from double_auction.value import *
from double_auction.auction import *
class fed:
    def __init__(self):
        self.args=args_parser()
        self.args.device = torch.device('cuda:{}'.format(self.args.gpu) if torch.cuda.is_available() and self.args.gpu != -1 else 'cpu')
        if args.dataset=='mnist':
            self.dataset_train=datasets.MNIST('../data/MNIST/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        elif args.dataset=='cifar':
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        elif args.dataset=='fmnist':
            self.dataset_train = datasets.FashionMNIST('../data/FMNIST/', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        elif args.dataset=='mpc':
            train_data = pd.read_csv("data/train.csv")
            num_of_classes = len(train_data["price_range"].unique())
            X = train_data.drop(["price_range"], axis=1)
            y = train_data["price_range"].values
            ct = make_column_transformer(
                (MinMaxScaler(),
                 ["battery_power", "clock_speed", "fc", "int_memory", "m_dep", "mobile_wt", "pc", "px_height",
                  "px_width",
                  "ram", "sc_h", "sc_w", "talk_time"]),
                (OneHotEncoder(handle_unknown="ignore"),
                 ["blue", "dual_sim", "four_g", "n_cores", "three_g", "touch_screen", "wifi"])
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=42)
            ct.fit(X_train)
            X_train_normal = ct.transform(X_train)
            self.dataset_train=X_train_normal
        else:
            exit('Error: unrecognized dataset')
        if args.dataset=='cifar' or args.dataset=='fmnist' or args.dataset=='mnist':
            self.dict_users, self.dict_users_test,self.distribute = dirichlet_split_noniid(self.dataset_train.train_labels, self.args.num_users)
        if args.dataset=='mnist' or args.dataset=='fmnist':
            if args.model=='mlp':
                self.net_0 = MLP(dim_in=784, dim_hidden=200, dim_out=args.num_classes).to(self.args.device)
                self.net=[MLP(dim_in=784, dim_hidden=200, dim_out=args.num_classes).to(self.args.device) for i in range(args.num_users)]
            elif args.model=='cnn':
                self.net_0=CNNMnist(args=args).to(self.args.device)
                self.net=[CNNMnist(args=args).to(self.args.device) for i in range(self.args.num_users)]
            else:
                exit('Error: unrecognized model')
        elif args.dataset=='cifar':
            if args.model=='cnn':
                self.net_0 = CNNCifar(args=args).to(self.args.device)
                self.net = [CNNCifar(args=args).to(self.args.device) for i in range(self.args.num_users)]
        self.w_locals=[self.net[i].state_dict() for i in range(self.args.num_users)]
        self.observation_space=self.args.num_users
        self.action_space=self.args.num_users
        self.action_space_low=-1
        self.action_space_high = 1
        self.ac=[]
        self.users = [user() for i in range(self.args.num_users)]
    def reset(self):
        self.args = args_parser()
        self.dict_users, self.dict_users_test, self.distribute = dirichlet_split_noniid(self.dataset_train.train_labels,
                                                                                        self.args.num_users)
        self.args.device = torch.device('cuda:{}'.format(self.args.gpu) if torch.cuda.is_available() and self.args.gpu != -1 else 'cpu')
        self.net_0 = MLP(dim_in=784, dim_hidden=200, dim_out=args.num_classes).to(self.args.device)
        self.net = [MLP(dim_in=784, dim_hidden=200, dim_out=args.num_classes).to(self.args.device) for i in range(args.num_users)]
        self.w_locals = [self.net[i].state_dict() for i in range(self.args.num_users)]
        self.ac=[]
        self.ac_=[]
        self.users = [user() for i in range(self.args.num_users)]
        for idx in range(self.args.num_users):
            local = LocalUpdate(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[idx],idxs_test=self.dict_users_test[idx])
            self.w_locals[idx]= local.train(net=copy.deepcopy(self.net[idx]).to(self.args.device))
            ac=local.test(self.net[idx])
            self.ac.append(ac)
        observation = [[0 for p in range(self.args.num_users)] for _ in range(self.args.num_users)]
        for j in range(self.args.num_users):
            for k in range(self.args.num_users):
                observation[j][k] = dif(self.w_locals[j], self.w_locals[k])
        return observation
    def step(self,action):
        self.ac_=[]
        for k in range(self.args.num_users):
            local = LocalUpdate(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[k],idxs_test=self.dict_users_test[k])
            w= local.train(net=copy.deepcopy(self.net[k]).to(self.args.device))
            ac= local.test(self.net[k])
            self.ac_.append(ac)
            self.w_locals[k] = copy.deepcopy(w)
        arr_all = [[0 for i in range(self.args.num_users)] for p in range(self.args.num_users)]
        reward=[0 for i in range(self.args.num_users)]
        l=0
        num=0
        M = [[0 for i in range(self.args.num_users)] for j in range(self.args.num_users)]
        for j in range(self.args.num_users):
            for k in range(self.args.num_users):
                arr_all[j][k] = dif(self.w_locals[j], self.w_locals[k])
            arr_sort = np.array(arr_all[j]).argsort()
            favor=[]
            if j< self.distribute[l] and l==0:
                num=self.distribute[l]
            elif j< self.distribute[l] and l>=1:
                num=self.distribute[l]-self.distribute[l-1]
            else:
                l+=1
                num=self.distribute[l]-self.distribute[l-1]
            if args.selection_method==0:
                _,num=torch.max(torch.tensor(action[j]),-1)
                print(num)
            for k in range(1, num+1):
                favor.append(arr_sort[args.num_users - k - 1].item())
                M[j][arr_sort[args.num_users - k - 1].item()]=1
        observation_ = [[0 for p in range(self.args.num_users)] for _ in range(self.args.num_users)]
        for j in range(self.args.num_users):
            for k in range(self.args.num_users):
                observation_[j][k] = dif(self.w_locals[j], self.w_locals[k])
        if args.selection_method==2:
            M = [[0 for i in range(self.args.num_users)] for j in range(self.args.num_users)]
        elif args.selection_method==3:
            M = [[1 for i in range(self.args.num_users)] for j in range(self.args.num_users)]
        if args.auction!=-1:
            M,self.users=trans(self.users,observation_,M,b_min_th=0,a_max_th=1e24)
        for j in range(args.num_users):
            favor=[]
            for k in range(args.num_users):
                if M[j][k]==1:
                    favor.append(k)
            self.net_0.load_state_dict(part_FedAvg(favor, self.w_locals, j))
            self.net[j]=copy.deepcopy(self.net_0)
            local = LocalUpdate(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[j],
                                idxs_test=self.dict_users_test[j])
            self.ac[j]= local.test(net=copy.deepcopy(self.net[j]).to(self.args.device))
            reward[j]=np.exp((self.ac[j]-96)*0.1)-1
        done = False

        observation_ = [[0 for p in range(self.args.num_users)] for _ in range(self.args.num_users)]
        for j in range(self.args.num_users):
            for k in range(self.args.num_users):
                observation_[j][k] = dif(self.w_locals[j], self.w_locals[k])
            observation_[j]=sorted(observation_[j],reverse=True)
        print(sum(self.ac)/len(self.ac))
        return observation_, reward, done,sum(self.ac)/len(self.ac)
