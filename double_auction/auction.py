import numpy as np
import random
from utils.options import args_parser
#一列是买方，一行是卖方
args=args_parser()
class Graph():
    def __init__(self,size):
        self.matrix_b=[[0 for j in range(size)]for i in range(size)]
        self.arr_a=None
        self.size=size
    def rando(self,size=args.num_users):
        self.matrix_b=[[random.randint(0,1) for j in range(size)]for i in range(size)]
        for i in range(size):
            self.matrix_b[i][i]=0
    def set_a(self,size=args.num_users):
        self.arr_a = [[random.uniform(0, 1) for j in range(size)] for i in range(size)]
    def set_b(self,size=args.num_users):
        self.matrix_b=[[random.randint(0,1) for j in range(size)] for i in range(size)]
    def print(self):
        for i in range(self.size):
            for j in range(self.size):
                print(self.matrix_b[i][j],end=' ')
            print('')
def partition(M):
    G1=[[0 for i in range(args.num_users)] for j in range(args .num_users)]
    for i in range(args.num_users):
        status=0
        su=0
        for j in range(args.num_users):
            a=M[i][j]
            if a!=0:
                su+=1
            if su==2:
                status=1
                break
        if status==1:
            G1[i]=M[i]
            M[i]=[0 for k in range(args.num_users)]
    G2=[[0 for i in range(args.num_users)] for j in range(args .num_users)]
    for i in range(args.num_users):
        status=0
        su=0
        for j in range(args.num_users):
            if M[j][i]!=0:
                su+=1
            if su==2:
                status=1
                break
        if status==1:
            for j in range(args.num_users):
                G2[j][i]=M[j][i]
            for j in range(args.num_users):
                M[j][i]=0
    G3=M

    return G1,G2,G3

def G2_auction(G2,observation,users,a_max_th=100000):#一个卖方多个买方
    for i in range(args.num_users):
        bid = [0 for j in range(args.num_users)]
        for j in range(args.num_users):
            if G2[j][i] == 1:
                bid[j] = users[j].buy(observation[j][i])
        b_min = min(bid)
        if b_min >= a_max_th:
            clearing_price = a_max_th
        else:
            clearing_price = b_min
        for k in range(args.num_users):
            if users[i].ask[k] == clearing_price:
                G2[i][k] = 0
        for j in range(args.num_users):
            if clearing_price <= bid[j] and clearing_price > users[i].ask[j] and G2[i][j] != 0:
                G2[i][j] = 1
                users[j].save -= clearing_price
                users[i].save += clearing_price
            else:
                G2[i][j] = 0
    return G2,users

def G1_auction(G1,observation,users,b_min_th=0):#一个买方多个卖方
    for i in range(args.num_users):
        bid=[0 for i in range(args.num_users)]
        for j in range(args.num_users):
            if G1[i][j]==1:
                bid[j]=users[i].buy(observation[i][j])
        a_max=max(users[i].ask)
        if a_max<=b_min_th:
            clearing_price=b_min_th
        else:
            clearing_price=a_max
        for k in range(args.num_users):
            if users[i].ask[k]==clearing_price:
                G1[i][k]=0
        for j in range(args.num_users):
            if clearing_price<=bid[j] and clearing_price>users[i].ask[j] and G1[i][j]!=0:
                G1[i][j]=1
                users[i].save-=clearing_price
                users[j].save+=clearing_price
            else:
                G1[i][j]=0
    return G1,users

def G3_auction(G3,users,observation,b_min_th=0,a_max_th=100000):

    bid=[]
    ask=[]
    bid_matrix=[[0 for i in range(args.num_users)]for j in range(args.num_users)]
    ask_matrix=[[0 for i in range(args.num_users)]for j in range(args.num_users)]
    for i in range(args.num_users):
        for j in range(args.num_users):
            if G3[i][j]==1:
                bid.append(users[i].buy(observation[i][j]))
                ask.append(users[j].ask[i])
                bid_matrix[i][j]=bid[len(bid)-1]
                ask_matrix[j][i]=users[j].ask[i]
    b_sort=np.argsort(bid)
    a_sort=np.argsort(ask)
    b_sort=b_sort[::-1]
    g=0
    if len(b_sort)!=0:
        for i in range(len(b_sort)):
            if a_sort[i] > b_sort[i]:
                break
            g += 1

        v = (bid[b_sort[g + 1]] + ask[a_sort[g + 1]]) / 2

        if ask[a_sort[g]] <= v and bid[b_sort[g]] >= v and g < len(b_sort):
            A = v
            B = v
        else:
            A = max(b_min_th, ask[a_sort[g]])
            B = min(a_max_th, bid[b_sort[g]])
        for i in range(args.num_users):
            for j in range(args.num_users):
                if bid_matrix[i][j] > B and ask_matrix[j][i] < A and G3[i][j] != 0:
                    users[i].save -= B
                    users[j].save += A
                else:
                    G3[i][j] = 0

    return G3,users

def auction(G1,G2,G3):
    for i in range(args.num_users):
        for j in range(args.num_users):
            if G1[i][j]==1 or G2[i][j]==1:
                G3[i][j]=1
    return G3

def trans(users,observation,M,b_min_th=0,a_max_th=100000):
    G1,G2,G3=partition(M)
    G1,users=G1_auction(G1,observation,users,b_min_th)
    G2,users=G2_auction(G2,observation,users,a_max_th)
    G3,users=G3_auction(G3,users,observation,b_min_th,a_max_th)
    M=auction(G1,G2,G3)
    for i in range(args.num_users):
        print(users[i].save,end=' ')

    print('交易后')
    for i in M:
        print(sum(i))
    return M,users
