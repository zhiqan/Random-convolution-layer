import torch
import torch.nn as nn



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

    def __init__(self, block, num_block, num_classes=10):
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
   
'''

计算参数与flops
'''
from thop import profile

input = torch.randn(1, 1, 1024)
macs, params = profile(model, inputs=(input, )) #flops为macs的一半




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

        mat = loadmat('D:\\论文mix-cnn\\模型\\西储大学轴承故障\\数据准\\训练集\\10类训_400.204_dataset1024.mat')
        mat1 = loadmat('D:\\论文mix-cnn\\模型\\西储大学轴承故障\\数据准\\测试集\\10类测_120.204_dataset1024.mat')
        
        #mat = loadmat('D:\\论文mix-cnn\\模型\\西储大学轴承故障\\数据\\10类_-10db_dataset1024.mat')
        #mat = loadmat('D:\\国际会议论文\会议论文资料\\PYshenduxuexi\\Fault_Diagnosis_CNN-master\\Datasets\\data7\\None\\dataset1024.mat')
        self.X_train = mat['X_train']
        self.X_test = mat1['X_train']
        self.y_train =np.array(mat['y_train'][:,0],dtype=int)
        self.y_test = np.array(mat1['y_train'][:,0],dtype=int)
        scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
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
噪声
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

testset = MyData(X_test, y_test1)
test_loader = DataLoader(dataset=testset,
                         batch_size=batch_size,
                        shuffle=True)
 
    
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



for epoch in range(30):
     print('epoch:',epoch)
     train()
     test()

torch.save(model, 'D:\\论文mix-cnn\\模型\\西储大学轴承故障\模型\\模型准\\1%shuju_98.61%模型对比DRSN-CW')

for epoch in range(10):
    X_train = data.X_train_minmax
    y_train = data.y_train
    X_test = data.X_test_minmax # 临时代替
    y_test = data.y_test
    #X_train, aaaa, y_train1, bbb= train_test_split(X_train, y_train, test_size=0.5)
    #X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.45)
    
    #X_train, aaaa, y_train1, bbb= train_test_split(X_train, y_train, test_size=0.9)
    #X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.1)
    
    #X_train, aaaa, y_train1, bbb= train_test_split(X_train, y_train, test_size=0.7)
    #X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.25)
    
    #X_train, aaaa, y_train1, bbb= train_test_split(X_train, y_train, test_size=0.96)
    #X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.084)
    X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.85)
    
    #X_train, aaaa, y_train1, bbb= train_test_split(X_train, y_train, test_size=0.98)
    #X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.024)
    
    testset = MyData(X_test, y_test1)
    test_loader = DataLoader(dataset=testset,
                             batch_size=batch_size,
                            shuffle=True)
    test()












'''
epochs设为100
'''

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

'''
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


#X_train=X_train1
#X_test=X_test1

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
for i in y3[0]:    
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


def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels
'''

mat = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\训练集\\750_30_2_beardataset.mat')
mat1 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\测试集\\230_30_2_beardataset.mat')
mat2 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\训练集\\750_30_2_geardataset.mat')
mat3 = loadmat('D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\数据准\\测试集\\230_30_2_geardataset.mat')
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
    if i==0:
        a.append(0)
    elif i==1:
        a.append(1)
    elif i==2:
        a.append(2)
    elif i==3:
        a.append(3)
    elif i==4:
        a.append(4)

for i in y2[0]:    
    if i==0 :
        a.append(5)
    elif i==1:       
        a.append(2)
    elif i==2:
        a.append(6)
    elif i==3:
        a.append(7)
    elif i==4:
        a.append(8)
 #测试集       
c=[]
for i in y3[0]:    
    if i==0:
        c.append(0)
    elif i==1:
        c.append(1)
    elif i==2:
        c.append(2)
    elif i==3:
        c.append(3)
    elif i==4:
        c.append(4)

for i in y4[0]:    
    if i==0:
        c.append(5)
    elif i==1:       
        c.append(2)
    elif i==2:
        c.append(6)
    elif i==3:
        c.append(7)
    elif i==4:
        c.append(8)


def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


y_test = c
y_train1 = a

scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
X_train_minmax = scaler.fit_transform(X_train.T).T
X_test_minmax = scaler.fit_transform(X_test.T).T    

X_train=X_train_minmax
X_test=X_test_minmax

X_va, X_test, y_va, y_test1= train_test_split(X_test, y_test, test_size=0.85)

X_train1, aaaa, y_train1, bbb= train_test_split(X_train, y_train1, test_size=0.96)
X_val, X_test, y_val, y_test1= train_test_split(X_test, y_test, test_size=0.064)

#X_train1, aaaa, y_train1, bbb= train_test_split(X_train, y_train1, test_size=0.98)
#X_val, X_test, y_val, y_test1= train_test_split(X_test, y_test, test_size=0.02)


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

    def __init__(self, block, num_block, num_classes=5):
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
        return torch.Tensor(np.array([self.pics[index]])), self.labels[index]

    def __len__(self):
        return len(self.pics)

    def get_tensors(self):
        return torch.Tensor(np.array([self.pics])), torch.Tensor(np.array(self.labels))
 
batch_size = 60
trainset = MyData(X_train, y_train1)
train_loader = DataLoader(dataset = trainset,
                         batch_size=batch_size,
                         shuffle=True)

testset = MyData(X_test, y_test)
test_loader = DataLoader(dataset=testset,
                         batch_size=batch_size,
                        shuffle=True)
 

    
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
           #torch.nn.init.xavier_normal_(Conv.weight)
           torch.nn.init.kaiming_normal_(Conv.weight)
        
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


for epoch in range(60):
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
           #torch.nn.init.xavier_normal_(Conv.weight)
           torch.nn.init.kaiming_normal_(Conv.weight)
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



#### 可视化 经过RCL的数据
for i,data in enumerate(train_loader):
      inputs,labels = data
      N, C, W = inputs.size()
      k = 17
      Conv = torch.nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=k//2, bias=False)
      torch.nn.init.kaiming_normal_(Conv.weight)
      inputs2 = Conv(inputs.reshape(-1, C, W)).reshape(N, C,  W)
      
       
       
       
       
a= inputs.numpy()

A=a[1].T

b= inputs2.detach().numpy()

b1=b[1]


scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
X_train_minmax = scaler.fit_transform(b1.T)


















o11 = torch.Tensor(X_test).unsqueeze(1)
with torch.no_grad():
    input1=o11
    out = model(input1)
    _, pre = torch.max(out.data, 1)


predictions = pre.numpy()

def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from scipy import interp
fpr = dict()
tpr = dict()
roc_auc = dict()
y_t = onehot(predictions)  
y_test11 = onehot(y_test1)  
classes=[0,1,2,3,4,5,6,7,8]

score = metrics.accuracy_score(y_test1, predictions)
print(f"Validation fold score(accuracy): {score}")



for i in range(len(classes)):
    fpr[i], tpr[i], thresholds = roc_curve(y_test11[:, i],y_t[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], thresholds = roc_curve(y_t.ravel(),y_test1.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# macro-average ROC curve 方法二）

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(len(classes)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# 求平均计算ROC包围的面积AUC
mean_tpr /= len(classes)
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
import matplotlib.pyplot as plt
from itertools import cycle
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(9), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))



plt.figure()
plt.plot(fpr["micro"], tpr["micro"],'k-',color='y',
         label='XXXX ROC curve micro-average(AUC = {0:0.4f})'
               ''.format(roc_auc["micro"]),
          linestyle='-.', linewidth=3)

plt.plot(fpr["macro"], tpr["macro"],'k-',color='k',
         label='XXXX ROC curve macro-average(AUC = {0:0.4f})'
               ''.format(roc_auc["macro"]),
          linestyle='-.', linewidth=3)
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc="lower right")
plt.grid(linestyle='-.')  
plt.grid(True)
plt.show()




















torch.save(model, 'D:\\论文mix-cnn\\模型\\东南大学齿轮箱\\模型\\20-0all_99.84%模型对比DRSN-CW')









