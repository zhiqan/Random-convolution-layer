# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 20:01:58 2022

@author: Owner
"""

import torch
import torch.nn as nn


class WDCNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=10,AdaBN=True):
        super(WDCNN, self).__init__()
 
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64,stride=16,padding=24),
            nn.BatchNorm1d(16,track_running_stats=AdaBN),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2,stride=2)
            )
 
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3,padding=1),
            nn.BatchNorm1d(32,track_running_stats=AdaBN),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))
 
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3,padding=1),
            nn.BatchNorm1d(64,track_running_stats=AdaBN),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  
 
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3,padding=1),
            nn.BatchNorm1d(64,track_running_stats=AdaBN),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1
 
        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64,track_running_stats=AdaBN),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # nn.AdaptiveMaxPool1d(4)
        )  
 
        self.fc=nn.Sequential(
            nn.Linear(64, 100),
            nn.BatchNorm1d(100,track_running_stats=AdaBN),
            nn.ReLU(inplace=True),
            nn.Linear(100, out_channel),
            nn.LogSoftmax(dim=1)
        )
 
    def forward(self, x):
        #print(x.shape)
        x = self.layer1(x) 
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = self.layer5(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
    
model = WDCNN()




from scipy import stats
from scipy.io import loadmat,savemat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import pandas as pd

import random
import numpy as np

#torch.save(model, 'D:\\论文mix-cnn\\模型\\西储大学轴承故障\\模型\\模型准\\4db_2hp_100%模型对比DRSN-CW')

class Data_read:
    def __init__(self, snr='None'):

        mat = loadmat('D:\\论文mix-cnn\\模型\\西储大学轴承故障\\数据准\\训练集\\10类训_400.204-4db_dataset1024.mat')
        mat1 = loadmat('D:\\论文mix-cnn\\模型\\西储大学轴承故障\\数据准\\测试集\\10类测_120.204-4db_dataset1024.mat')
        
        #mat = loadmat('D:\\论文mix-cnn\\模型\\西储大学轴承故障\\数据\\10类_-10db_dataset1024.mat')
        #mat = loadmat('D:\\国际会议论文\会议论文资料\\PYshenduxuexi\\Fault_Diagnosis_CNN-master\\Datasets\\data7\\None\\dataset1024.mat')
        self.X_train = mat['X_train']
        self.X_test = mat1['X_train']
        self.y_train =np.array(mat['y_train'][:,0],dtype=int)
        self.y_test = np.array(mat1['y_train'][:,0],dtype=int)
        scaler = MinMaxScaler()
        self.X_train_minmax = scaler.fit_transform(self.X_train.T).T
        self.X_test_minmax = scaler.fit_transform(self.X_test.T).T

    #def add_noise(self,snr):



if __name__ == '__main__':
    Data_read()

'Two_wdcnn训练前准备'
data = Data_read(snr='None')
'''
# 选择一组训练与测试集
X_train = data.X_train_minmax # 临时代替
y_train = data.y_train

# 各组测试集
X_test = data.X_test_minmax
y_test = data.y_test

X_test = np.vstack((data.X_test_minmax,data.X_train_minmax))
y_test = np.vstack((data.y_test,data.y_train))
'''
'''
比例

X_train = data.X_train_minmax
y_train = data.y_train
X_test = data.X_test_minmax # 临时代替
y_test = data.y_test
#X_train, aaaa, y_train1, bbb= train_test_split(X_train, y_train, test_size=0.5)
#X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.45)

X_train, aaaa, y_train1, bbb= train_test_split(X_train, y_train, test_size=0.9)
X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.1)

#X_train, aaaa, y_train1, bbb= train_test_split(X_train, y_train, test_size=0.7)
#X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.25)

#X_train, aaaa, y_train1, bbb= train_test_split(X_train, y_train, test_size=0.96)
#X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.084)
    
X_train, aaaa, y_train1, bbb= train_test_split(X_train, y_train, test_size=0.98)
X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.024)


X_train, aaaa, y_train1, bbb= train_test_split(X_train, y_train, test_size=0.99)
X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.015)

#x_train1=X_train[:,np.newaxis,:]#转换成cnn的输入格式
#x_va=X_va[:,:,np.newaxis]
#x_test1=X_test[:,np.newaxis,:]
'''

'''
'''

X_train = data.X_train_minmax
y_train1 = data.y_train
X_test = data.X_test_minmax # 临时代替
y_test = data.y_test

X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.85)


from torch import optim
from torch.utils.data import DataLoader

torch.manual_seed(1)  # 设置随机种子；可复现性
# 超参数
LR = 0.001
crossEntropyloss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),LR)


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
trainset = MyData(X_train, y_train1)
train_loader = DataLoader(dataset = trainset,
                         batch_size=batch_size,
                         shuffle=True)

testset = MyData(X_test, y_test)
test_loader = DataLoader(dataset=testset,
                         batch_size=batch_size,
                        shuffle=True)




'''

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



for epoch in range(80):
     print('epoch:',epoch)
     train()
     test()
'''

    
   
    
   

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



for epoch in range(150):
     print('epoch:',epoch)
     train()
     test()    
   
    
   
    
   
    
   
    
   
    