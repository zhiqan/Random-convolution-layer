# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:05:08 2021

@author: Administrator
"""
import numpy as np
import torch
import os
import re
import scipy.io as scio
import scipy.signal
from torch.utils import data as da
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



import os
import random
import numpy as np
import pandas as pd
import heapq
from scipy import stats
from scipy.io import loadmat,savemat
from sklearn.model_selection import train_test_split

hp_rpm = {'0':'1797','1':'1772','2':'1750','3':'1730'}
# 速度与负载马力的对应关系

class CWRU_Data:
    def __init__(self, length, m, snr=None, FE=0,files=0):
        # root directory of all data
        rdir = os.path.join('./Datasets/CWRU')
        all_lines = open('./Datasets/metadata{}.txt'.format(files)).readlines()
        lines = []
        for line in all_lines:
            l = line.split()
            lines.append(l)

        self.length = length  # sequence length
        self.samples = m # sample number
        self.snr = snr
        self._slice_data(rdir, lines, FE)
        # shuffle training and test arrays
        #self._shuffle()
        self.labels = set(line[1] for line in lines)
        self.nclasses = len(self.labels)  # number of classes

    def _slice_data(self, rdir, infos, FE):
        #self.X_train = np.zeros((0, self.length))
        #self.X_test = np.zeros((0, self.length))
        #self.y_train = []
        #self.y_test = []
        if not FE:
            self.X = np.zeros((0, self.length))
            self.y = np.zeros((0, 1))
        else:
            self.X = np.zeros((0, 15))
            self.y = np.zeros((0, 1))
        for idx, info in enumerate(infos):
            # directory of this file
            fdir = os.path.join(rdir, info[0])
            fpath = os.path.join(fdir, info[1] + '.mat')
            mat_dict = loadmat(fpath)
            # key = filter(lambda x: 'DE_time' in x, mat_dict.keys())[0]
            filter_i = filter(lambda x: 'DE_time' in x, mat_dict.keys())
            filter_list = [item for item in filter_i]
            key = filter_list[0]
            time_series = mat_dict[key][:, 0]
            # 原始信号中添加高斯白噪声
            time_series = self.add_wgn(time_series)
            # 重叠采样获得训练片段样本
            samples = self.samples
            for sample in range(samples):
                start = (3*sample*self.length)//10
                segment = time_series[start : start+self.length]
                if not FE:
                    feat = segment
                else:
                    feat = self.feat_extract(segment)
                self.X = np.vstack((self.X, feat))
                self.y = np.vstack((self.y, idx%10))

        self.X_train, self.X_test, self.y_train, self.y_test\
            = train_test_split(self.X, self.y, test_size=1000)

    # 添加高斯白噪声
    def add_wgn(self, x):
        if self.snr is not None:
            snr = 10 ** (self.snr / 10.0)
            xpower = np.sum(x ** 2) / len(x)
            npower = xpower / snr
            return x + np.random.randn(len(x)) * np.sqrt(npower)
        return x

    def feat_extract(self, x):
        feat = pd.DataFrame(index=[0])
        feat['mean'] = np.mean(x)
        feat['std'] = np.std(x)
        feat['max'] = np.max(x)
        feat['min'] = np.min(x)
        feat['peak'] = np.max(np.abs(x))
        feat['p2p'] = np.max(x) - np.min(x)
        feat['rms'] = np.sqrt(np.mean(np.square(x)))
        feat['smr'] = np.square(np.mean(np.sqrt(abs(x))))
        feat['ma'] = np.mean(abs(x))
        feat['kurt'] = stats.kurtosis(x)
        feat['skew'] = stats.skew(x)
        feat['shape_f'] = feat['rms'] / feat['ma']
        feat['crest_f'] = feat['peak'] / feat['rms']
        feat['impulse_f'] = feat['peak'] / feat['ma']
        feat['clear_f'] = feat['peak'] / feat['smr']

        return feat


"""
    def _shuffle(self):
        # shuffle training samples
        
        index = list(range(self.X_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = tuple(self.y_train[i] for i in index)

        # shuffle test samples
        index = list(range(self.X_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_test = self.X_test[index]
        self.y_test = tuple(self.y_test[i] for i in index)
"""

## 处理获得用于学习训练的数据，存为mat文件方便下次使用
    #snr_list = [-6, -4, -2, 0, 2, 4, 6, None]
data = CWRU_Data(length=1024, m=100, snr=-6, FE=0,files=10)

#savemat('..\\Datasets\\data7\\0db\\dataset1024.mat', {'X_train': data.X_train,
                                                #'X_test': data.X_test,
                                                #'y_train': data.y_train})



X_train = data.X_train
X_test = data.X_test
y_train=data.y_train[:,0].astype("int32")
y_test=data.y_test[:,0].astype("int32")


ss = MinMaxScaler()
X_train = X_train.T
X_test =X_test.T

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
