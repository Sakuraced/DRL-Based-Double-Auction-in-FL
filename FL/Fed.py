#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import math
import numpy as np
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
def dif(w1,w2):
    sum=0
    sum1=0
    for k in w1.keys():
        sum=abs(w1[k]-w2[k])
    for i in sum:
        sum1+=i*i
    a=sum1.item()
    return math.exp(-math.sqrt(a)/0.05)
def part_FedAvg(favor,w,j):
    w0=copy.deepcopy(w[j])
    for k in w[j].keys():
        for m in favor:
            w0[k]+=w[m][k]
        w0[k]=torch.div(w0[k],(len(favor)+1))
    return w0

def FedAMP(favor,w,j):
    sum=0
    for m in favor:
        temp=dif(w[j],w[m])
        sum+=temp
        for k in w[j].keys():
            w[j][k]+=w[m][k]*temp
    for k in w[j].keys():
        w[j][k]=w[j][k]/(sum+1)
    return w[j]





