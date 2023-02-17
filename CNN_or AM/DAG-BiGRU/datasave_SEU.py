# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 16:38:50 2023

@author: Owner
"""

import torch
from scipy.io import loadmat

import numpy as np

from torch.utils import data as da
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split




mat = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\训练集\\750_beardataset.mat')
mat1 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\测试集\\230_beardataset.mat')
mat2 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\训练集\\750_geardataset.mat')
mat3 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\测试集\\230_geardataset.mat')
#mat = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据\\轴承数据\\7千5beardatasetz方向.mat')
#mat1 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\数据\\齿轮数据\\7千5geardatasetz方向.mat')

X_train1= mat['X_train']
X_train2= mat2['X_train']
y1=mat['y_train']
y2=mat2['y_train']


X_test1= mat1['X_test']
X_test2= mat3['X_test']
y3=mat1['y_test']
y4=mat3['y_test']

X_train=np.vstack((X_train1,X_train2)) 
X_test=np.vstack((X_test1,X_test2))
a=[]
for i in y1[0]:    
    if i==0 or i==1:
        a.append(0)
    elif i==2 or i==3:
        a.append(1)
    elif i==4 or i==5:
        a.append(2)
    elif i==6 or i==7:
        a.append(3)
    elif i==8 or i==9:
        a.append(4)

for i in y2[0]:    
    if i==0 or i==1:
        a.append(5)
    elif i==2 or i==3:       
        a.append(2)
    elif i==4 or i==5:
        a.append(6)
    elif i==6 or i==7:
        a.append(7)
    elif i==8 or i==9:
        a.append(8)
 #测试集       
c=[]
for i in y4[0]:    
    if i==0 or i==1:
        c.append(0)
    elif i==2 or i==3:
        c.append(1)
    elif i==4 or i==5:
        c.append(2)
    elif i==6 or i==7:
        c.append(3)
    elif i==8 or i==9:
        c.append(4)

for i in y4[0]:    
    if i==0 or i==1:
        c.append(5)
    elif i==2 or i==3:       
        c.append(2)
    elif i==4 or i==5:
        c.append(6)
    elif i==6 or i==7:
        c.append(7)
    elif i==8 or i==9:
        c.append(8)



ss = MinMaxScaler()
X_train = X_train2.T
X_test =X_test2.T

y_train= np.array(a).astype("int32")
y_test= np.array(c).astype("int32")


X_train = ss.fit_transform(X_train).T
X_test= ss.fit_transform(X_test).T

X_train = torch.from_numpy(X_train).unsqueeze(1)
X_test = torch.from_numpy(X_test).unsqueeze(1)
class TrainDataset(da.Dataset):
    def __init__(self):
        self.Data = X_train
        self.Label = y_train
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)
class TestDataset(da.Dataset):
    def __init__(self):
        self.Data = X_test
        self.Label = y_test
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)
Train = TrainDataset()
Test = TestDataset()
train_loader = da.DataLoader(Train, batch_size=128, shuffle=True)
test_loader = da.DataLoader(Test, batch_size=10, shuffle=False) 




mat = loadmat('D:\\参考文献\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\FSM3\\PU\\PU_N15_M01_F10.mat')
mat1 = loadmat('D:\\参考文献\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\FSM3\\PU\\PU_N15_M07_F04.mat')
mat2 = loadmat('D:\\参考文献\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\FSM3\\PU\\PU_N15_M07_F10.mat')

X_train= mat2['X_train']
X_train1= mat1['X_train']
X_train2= mat2['X_train']

y_train=mat2['y_train'][0,:]
y_train1=mat1['y_train'][0,:]
y_train2=mat2['y_train'][0,:]

X_train=np.vstack((X_train,X_train1,X_train2)) 
y_train=np.hstack((y_train,y_train1,y_train2)) 



scaler = MinMaxScaler()
X_train_minmax = scaler.fit_transform(X_train.T).T

X_train=X_train_minmax


X_test, X_train1, y_test, y_train1= train_test_split(X_train, y_train, test_size=0.75)


X_train1 = torch.from_numpy(X_train1).unsqueeze(1)
X_test1 = torch.from_numpy(X_test).unsqueeze(1)


y_train=  y_train1.astype("int32")
y_test= y_test.astype("int32")




class TrainDataset(da.Dataset):
    def __init__(self):
        self.Data = X_train1
        self.Label = y_train1
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)
class TestDataset(da.Dataset):
    def __init__(self):
        self.Data = X_test1
        self.Label = y_test
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)
Train = TrainDataset()
Test = TestDataset()
train_loader = da.DataLoader(Train, batch_size=50, shuffle=True)
test_loader = da.DataLoader(Test, batch_size=10, shuffle=False) 









import os

mat = loadmat('D:\\参考文献\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\FSM3\\HIT\\800rpm.mat')
mat1 = loadmat('D:\\参考文献\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\FSM3\\HIT\\1000rpm.mat')
mat2 = loadmat('D:\\参考文献\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\FSM3\\HIT\\1200rpm.mat')

X_train= mat2['X_train']
X_train1= mat1['X_train']
X_train2= mat2['X_train']

y_train=mat2['y_train'][0,:]
y_train1=mat1['y_train'][0,:]
y_train2=mat2['y_train'][0,:]


X_train=np.vstack((X_train,X_train1,X_train2)) 
y_train=np.hstack((y_train,y_train1,y_train2)) 


scaler = MinMaxScaler()
X_train_minmax = scaler.fit_transform(X_train.T).T

X_train=X_train_minmax


X_test, X_train, y_test, y_train1= train_test_split(X_train, y_train, test_size=0.75)


X_train1 = torch.from_numpy(X_train).unsqueeze(1)
X_test1 = torch.from_numpy(X_test).unsqueeze(1)

y_train=  y_train1.astype("int32")
y_test= y_test.astype("int32")


class TrainDataset(da.Dataset):
    def __init__(self):
        self.Data = X_train1
        self.Label = y_train
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)
class TestDataset(da.Dataset):
    def __init__(self):
        self.Data = X_test1
        self.Label = y_test
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)
Train = TrainDataset()
Test = TestDataset()

train_loader = da.DataLoader(Train, batch_size=50, shuffle=True)

test_loader = da.DataLoader(Test, batch_size=10, shuffle=False) 

























