#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 22:43:08 2021

@author: zhoukuan
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import c_
"""
x = list(range(1, 21))  # epoch array
loss = [2 / (i**2) for i in x]  # loss values array
plt.ion()
for i in range(1, len(x)):
    ix = x[:i]
    iy = loss[:i]
    plt.cla()
    plt.title("loss")
    plt.plot(ix, iy)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.pause(0.5)
plt.ioff()
plt.show()
"""

#sam = np.array([[1,2],[2,3],[2,5],[3,1],[3,6],[4,5],[5,1],[5,3]]) #设定初始的点位

xt = np.arange(1,9)
e = np.random.randn(xt.size)
yt = xt + 1 + e
t = np.array([1,1])
sam = scipy.c_[xt,yt]

#改进：
#总数据多，选参数少
#变步长

#根据上面的公式分别求出Q对B的偏导、变化后的B、残差平方和Q
def deltaB(sam, B_init, lr=0.01):
    X = scipy.c_[np.ones(sam.shape[0]),sam[:,0]]
    Y = sam[:,1]
    Q = np.dot((Y - np.dot(X,B_init)).T,Y - np.dot(X,B_init))
    Q_B = -np.dot(X.T,Y) + np.dot(np.dot(X.T,X),B_init)
    B = B_init - lr * Q_B
    
    return Q_B, B, Q

iter_num = 400
B_init = np.array([8,-1.2])
#以下三个_his用来存放每次生成的值
B_his = B_init
Q_B_his = np.array([0,0])
Q_his = []
for i in range(iter_num):
    sel = np.random.choice(sam.shape[0],6,replace=False) #随机挑选点
    Q_B,B,Q = deltaB(sam[sel], B_his.reshape(2,-1)[:,i])
    Q_B_his = scipy.c_[Q_B_his,Q_B]
    B_his = np.round(scipy.c_[B_his,B],4)
    Q_his.append(Q.tolist())

x=np.array([0,10])
plt.ion()
for j in range(0,B_his.shape[1]):
    plt.cla()    
    #y = scipy.c_[y,B_his[0,j]+B_his[1,j]*x] #把每个B对应的y放到一起
    plt.plot(x,B_his[0,j]+B_his[1,j]*x)
    plt.scatter(sam[:,0], sam[:,1])
    plt.xlabel("x")
    plt.ylabel("y")
    if j<=5:
        plt.pause(2)
    else:
        plt.pause(0.01)
    plt.xlim(0,10)
    plt.ylim(0,15)
plt.ioff()
plt.show()
