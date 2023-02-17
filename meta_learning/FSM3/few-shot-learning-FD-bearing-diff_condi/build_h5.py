# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 23:08:34 2022

@author: Owner
"""

import h5py
import numpy as np


import os
import pandas as pd

import random
import numpy as np
import heapq
from scipy import stats
from scipy.io import loadmat,savemat
#




mat = loadmat('D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\XICHU\\0HP.mat')

mat1 = loadmat('D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\XICHU\\1HP.mat')

mat2 = loadmat('D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\XICHU\\3HP.mat')
#mat3 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\测试集\\230_30_2_geardataset.mat')

#mat2 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\训练集\\750_20_0_geardataset.mat')
#mat3 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\测试集\\230_30_2_geardataset.mat')

#mat = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据\\轴承数据\\7千5beardatasetz方向.mat')
#mat1 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\数据\\齿轮数据\\7千5geardatasetz方向.mat')

X_train= mat['X_train']
X_train1= mat1['X_train']
X_train2= mat2['X_train']

y1=mat['y_train'].T
y2=mat1['y_train'].T
y3=mat2['y_train'].T


 
f=h5py.File("D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\XICHU\\0HP.h5","w")
 


d1=f.create_dataset("X",data=X_train)
d2=f.create_dataset("Y",data=y1)
f.close()


f=h5py.File("D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\XICHU\\1HP.h5","w")
 

d1=f.create_dataset("X",data=X_train1)
d2=f.create_dataset("Y",data=y2)
f.close()



f=h5py.File("D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\XICHU\\3HP.h5","w")
 

d1=f.create_dataset("X",data=X_train2)
d2=f.create_dataset("Y",data=y3)
f.close()









mat = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\训练集\\750_20_0_beardataset.mat')
mat2 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\训练集\\750_30_2_beardataset.mat')




mat = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\训练集\\750_20_0_geardataset.mat')
mat2 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\训练集\\750_30_2_geardataset.mat')
#mat3 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\测试集\\230_30_2_geardataset.mat')

#mat2 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\训练集\\750_20_0_geardataset.mat')
#mat3 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\测试集\\230_30_2_geardataset.mat')

#mat = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据\\轴承数据\\7千5beardatasetz方向.mat')
#mat1 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\数据\\齿轮数据\\7千5geardatasetz方向.mat')

X_train1= mat['X_train']
X_train2= mat2['X_train']
y1=mat['y_train']
y2=mat2['y_train']


 
f=h5py.File("D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\SEU_gear\\750_20_0.h5","w")
 


d1=f.create_dataset("X",data=X_train1)
d2=f.create_dataset("Y",data=y1)
f.close()


f=h5py.File("D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\SEU_gear\\750_30_2.h5","w")
 


d1=f.create_dataset("X",data=X_train2)
d2=f.create_dataset("Y",data=y2)
f.close()



'''
f = h5py.File('D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\SEU_bear\\750_30_2.h5','r') 

X_orig = np.array(f['X'][:])
Y_orig = np.array(pd.read_hdf(data_file, key='Y'))


Y_orig  = np.array(f['Y'][:][0])

'''

y_support_onehot = torch.zeros((5*10, 5)).scatter_(1, Y_orig[50], 1)



x = x.unsqueeze(1).float()
y = y.squeeze(1).long()

loss_list, accu_list = [], []

for i, (x, y) in enumerate(data_loader):
    if i<2:
        x1=x
        y1=y
        
###############################
#PU
###########################


mat = loadmat('D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\PU\\PU_N15_M01_F10.mat')

mat1 = loadmat('D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\PU\\PU_N15_M07_F04.mat')

mat2 = loadmat('D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\PU\\PU_N15_M07_F10.mat')
#mat3 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\测试集\\230_30_2_geardataset.mat')

#mat2 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\训练集\\750_20_0_geardataset.mat')
#mat3 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\测试集\\230_30_2_geardataset.mat')

#mat = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据\\轴承数据\\7千5beardatasetz方向.mat')
#mat1 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\数据\\齿轮数据\\7千5geardatasetz方向.mat')

X_train= mat['X_train']
X_train1= mat1['X_train']
X_train2= mat2['X_train']

y1=mat['y_train']
y2=mat1['y_train']
y3=mat2['y_train']


 
f=h5py.File("D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\PU\\PU_N15_M01_F10.h5","w")
 


d1=f.create_dataset("X",data=X_train)
d2=f.create_dataset("Y",data=y1)
f.close()


f=h5py.File("D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\PU\\PU_N15_M07_F04.h5","w")
 

d1=f.create_dataset("X",data=X_train1)
d2=f.create_dataset("Y",data=y2)
f.close()



f=h5py.File("D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\PU\\PU_N15_M07_F10.h5","w")
 

d1=f.create_dataset("X",data=X_train2)
d2=f.create_dataset("Y",data=y3)
f.close()







###############################
#HIT
###########################


mat = loadmat('D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\HIT\\800rpm.mat')

mat1 = loadmat('D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\HIT\\1000rpm.mat')

mat2 = loadmat('D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\HIT\\1200rpm.mat')
#mat3 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\测试集\\230_30_2_geardataset.mat')

#mat2 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\训练集\\750_20_0_geardataset.mat')
#mat3 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\测试集\\230_30_2_geardataset.mat')

#mat = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据\\轴承数据\\7千5beardatasetz方向.mat')
#mat1 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\数据\\齿轮数据\\7千5geardatasetz方向.mat')

X_train= mat['X_train']
X_train1= mat1['X_train']
X_train2= mat2['X_train']

y1=mat['y_train']
y2=mat1['y_train']
y3=mat2['y_train']


 
f=h5py.File("D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\HIT\\HIT_800RPM.h5","w")
 


d1=f.create_dataset("X",data=X_train)
d2=f.create_dataset("Y",data=y1)
f.close()


f=h5py.File("D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\HIT\\HIT_1000RPM.h5","w")
 

d1=f.create_dataset("X",data=X_train1)
d2=f.create_dataset("Y",data=y2)
f.close()



f=h5py.File("D:\\参考文献\\元学习故障诊断-Few-shot-Learning-for-Fault-Diagnosis-main\\FSM3\\HIT\\HIT_1200RPM.h5","w")
 

d1=f.create_dataset("X",data=X_train2)
d2=f.create_dataset("Y",data=y3)
f.close()




























