# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:11:47 2023

@author: Owner
"""

from scipy import stats
from scipy.io import loadmat,savemat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import pandas as pd

import random
import numpy as np

import torch
import torch.nn as nn


from scipy import stats
from scipy.io import loadmat,savemat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import pandas as pd

import random
import numpy as np
mat = loadmat('D:\\参考文献\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\FSM3\\PU\\PU_N15_M01_F10.mat')
mat1 = loadmat('D:\\参考文献\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\FSM3\\PU\\PU_N15_M07_F04.mat')
mat2 = loadmat('D:\\参考文献\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\FSM3\\PU\\PU_N15_M07_F10.mat')

X_train= mat['X_train']
X_train1= mat1['X_train']
X_train2= mat2['X_train']

y_train=mat['y_train'][0,:]
y_train1=mat1['y_train'][0,:]
y_train2=mat2['y_train'][0,:]

X_train=np.vstack((X_train,X_train1,X_train2)) 
y_train=np.hstack((y_train,y_train1,y_train2)) 



scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
X_train_minmax = scaler.fit_transform(X_train.T).T

X_train=X_train_minmax


X_test, X_train1, y_test, y_train1= train_test_split(X_train, y_train, test_size=0.75)





from torch import optim
from torch.utils.data import DataLoader

from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self, pics, labels):
        self.pics = pics
        self.labels = labels

        # print(len(self.pics.files))
        # print(len(self.labels.files))

    def __getitem__(self, index):
        # print(index)
        # print(len(self.pics))
        assert index < len(self.pics)
        return torch.Tensor([self.pics[index]]), self.labels[index]

    def __len__(self):
        return len(self.pics)

    def get_tensors(self):
        return torch.Tensor([self.pics]), torch.Tensor(self.labels)
 
batch_size = 80
trainset = MyData(X_train1, y_train1)
train_loader = DataLoader(dataset = trainset,
                         batch_size=batch_size,
                         shuffle=True)

testset = MyData(X_test, y_test)
test_loader = DataLoader(dataset=testset,
                         batch_size=batch_size,
                        shuffle=True)
 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels, gap_size=(1))
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion),
            self.shrinkage
        )
        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):

         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
        # a = self.residual_function(x),
        # b = self.shortcut(x),
        # c = a+b
        # return c


class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = gap_size
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = nn.AdaptiveAvgPool1d(self.gap)(x)
        x = torch.flatten(x, 1)
        #average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class RSNet(nn.Module):

    def __init__(self, block, num_block, num_classes=3):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make rsnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual shrinkage block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a rsnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def rsnet18():
    """ return a RSNet 18 object
    """
    return RSNet(BasicBlock, [2, 2, 2, 2])


def rsnet34():
    """ return a RSNet 34 object
    """
    return RSNet(BasicBlock, [3, 4, 6, 3])

if __name__=='__main__':
    model = rsnet18()





torch.manual_seed(1)  # 设置随机种子；可复现性
# 超参数
LR = 0.001
crossEntropyloss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),LR)




    
def train():
   # 训练状态
   model.train()
   for i,data in enumerate(train_loader):
       inputs,labels = data
       out = model(inputs)
       loss = crossEntropyloss(out,labels.long())
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
   
   correct = 0
   for i,data in enumerate(train_loader):
       inputs,labels = data
       out = model(inputs)
       _,predictions = torch.max(out,1)
       correct +=(predictions == labels).sum()
   print("Train acc:{0}".format(correct.item()/len(trainset)))
        
def test():
    model.eval()
    correct = 0
    for i,data in enumerate(test_loader):
        inputs,labels = data
        out = model(inputs)
        _,predictions = torch.max(out,1)
        correct +=(predictions == labels).sum()
    print("Test acc:{0}".format(correct.item()/len(testset)))



for epoch in range(20):
     print('epoch:',epoch)
     train()
     test()
     
     


def train():
   # 训练状态
   model.train()
   for i,data in enumerate(train_loader):
       inputs,labels = data
       N, C, W = inputs.size()
       p = np.random.rand()
       K = [1, 3, 5, 7, 11, 15]
       if p > 0.5:
           k = K[np.random.randint(0, len(K))]
           Conv = torch.nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=k//2, bias=False)
           torch.nn.init.xavier_normal_(Conv.weight)
           inputs = Conv(inputs.reshape(-1, C, W)).reshape(N, C,  W)
       out = model(inputs)
       loss = crossEntropyloss(out,labels.long())
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
   
   correct = 0
   for i,data in enumerate(train_loader):
       inputs,labels = data
       #N, C, W = inputs.size()
       #p = np.random.rand()
       #K = [1, 3, 5, 7, 11, 15]
       #if p > 0.5:
           #k = K[np.random.randint(0, len(K))]
           #Conv = torch.nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=k//2, bias=False)
           #torch.nn.init.xavier_normal_(Conv.weight)
           #inputs = Conv(inputs.reshape(-1, C, W)).reshape(N, C,  W)
       
       out = model(inputs)
       _,predictions = torch.max(out,1)
       correct +=(predictions == labels).sum()
   print("Train acc:{0}".format(correct.item()/len(trainset)))
        
def test():
    model.eval()
    correct = 0
    for i,data in enumerate(test_loader):
        inputs,labels = data
        out = model(inputs)
        _,predictions = torch.max(out,1)
        correct +=(predictions == labels).sum()
    print("Test acc:{0}".format(correct.item()/len(testset)))


for epoch in range(30):
     print('epoch:',epoch)
     train()
     test()






















##########################################
#########################
########################################  HITHIT  ####################


import os

mat = loadmat('D:\\参考文献\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\FSM3\\HIT\\800rpm.mat')
mat1 = loadmat('D:\\参考文献\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\FSM3\\HIT\\1000rpm.mat')
mat2 = loadmat('D:\\参考文献\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\FSM3\\HIT\\1200rpm.mat')

X_train= mat1['X_train']
X_train1= mat1['X_train']
X_train2= mat2['X_train']

y_train=mat1['y_train'][0,:]
y_train1=mat1['y_train'][0,:]
y_train2=mat2['y_train'][0,:]

X_train=np.vstack((X_train,X_train1,X_train2)) 
y_train=np.hstack((y_train,y_train1,y_train2)) 



scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
X_train_minmax = scaler.fit_transform(X_train.T).T

X_train=X_train_minmax


X_test, X_train1, y_test, y_train1= train_test_split(X_train, y_train, test_size=0.75)





from torch import optim
from torch.utils.data import DataLoader

from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self, pics, labels):
        self.pics = pics
        self.labels = labels

        # print(len(self.pics.files))
        # print(len(self.labels.files))

    def __getitem__(self, index):
        # print(index)
        # print(len(self.pics))
        assert index < len(self.pics)
        return torch.Tensor([self.pics[index]]), self.labels[index]

    def __len__(self):
        return len(self.pics)

    def get_tensors(self):
        return torch.Tensor([self.pics]), torch.Tensor(self.labels)
 
batch_size = 80
trainset = MyData(X_train1, y_train1)
train_loader = DataLoader(dataset = trainset,
                         batch_size=batch_size,
                         shuffle=True)

testset = MyData(X_test, y_test)
test_loader = DataLoader(dataset=testset,
                         batch_size=batch_size,
                        shuffle=True)
 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels, gap_size=(1))
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion),
            self.shrinkage
        )
        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):

         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
        # a = self.residual_function(x),
        # b = self.shortcut(x),
        # c = a+b
        # return c


class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = gap_size
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = nn.AdaptiveAvgPool1d(self.gap)(x)
        x = torch.flatten(x, 1)
        #average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class RSNet(nn.Module):

    def __init__(self, block, num_block, num_classes=6):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make rsnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual shrinkage block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a rsnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def rsnet18():
    """ return a RSNet 18 object
    """
    return RSNet(BasicBlock, [2, 2, 2, 2])


def rsnet34():
    """ return a RSNet 34 object
    """
    return RSNet(BasicBlock, [3, 4, 6, 3])

if __name__=='__main__':
    model = rsnet18()





torch.manual_seed(1)  # 设置随机种子；可复现性
# 超参数
LR = 0.001
crossEntropyloss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),LR)




    
def train():
   # 训练状态
   model.train()
   for i,data in enumerate(train_loader):
       inputs,labels = data
       out = model(inputs)
       loss = crossEntropyloss(out,labels.long())
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
   
   correct = 0
   for i,data in enumerate(train_loader):
       inputs,labels = data
       out = model(inputs)
       _,predictions = torch.max(out,1)
       correct +=(predictions == labels).sum()
   print("Train acc:{0}".format(correct.item()/len(trainset)))
        
def test():
    model.eval()
    correct = 0
    for i,data in enumerate(test_loader):
        inputs,labels = data
        out = model(inputs)
        _,predictions = torch.max(out,1)
        correct +=(predictions == labels).sum()
    print("Test acc:{0}".format(correct.item()/len(testset)))



for epoch in range(20):
     print('epoch:',epoch)
     train()
     test()
     
     


def train():
   # 训练状态
   model.train()
   for i,data in enumerate(train_loader):
       inputs,labels = data
       N, C, W = inputs.size()
       p = np.random.rand()
       K = [1, 3, 5, 7, 11, 15]
       if p > 0.5:
           k = K[np.random.randint(0, len(K))]
           Conv = torch.nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=k//2, bias=False)
           torch.nn.init.xavier_normal_(Conv.weight)
           inputs = Conv(inputs.reshape(-1, C, W)).reshape(N, C,  W)
       out = model(inputs)
       loss = crossEntropyloss(out,labels.long())
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
   
   correct = 0
   for i,data in enumerate(train_loader):
       inputs,labels = data
       #N, C, W = inputs.size()
       #p = np.random.rand()
       #K = [1, 3, 5, 7, 11, 15]
       #if p > 0.5:
           #k = K[np.random.randint(0, len(K))]
           #Conv = torch.nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=k//2, bias=False)
           #torch.nn.init.xavier_normal_(Conv.weight)
           #inputs = Conv(inputs.reshape(-1, C, W)).reshape(N, C,  W)
       
       out = model(inputs)
       _,predictions = torch.max(out,1)
       correct +=(predictions == labels).sum()
   print("Train acc:{0}".format(correct.item()/len(trainset)))
        
def test():
    model.eval()
    correct = 0
    for i,data in enumerate(test_loader):
        inputs,labels = data
        out = model(inputs)
        _,predictions = torch.max(out,1)
        correct +=(predictions == labels).sum()
    print("Test acc:{0}".format(correct.item()/len(testset)))


for epoch in range(30):
     print('epoch:',epoch)
     train()
     test()























