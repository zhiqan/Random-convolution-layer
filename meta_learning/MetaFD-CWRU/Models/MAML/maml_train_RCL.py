"""
Code refers to: https://github.com/learnables/learn2learn/blob/master/examples/vision/maml_omniglot.py
By Yancy F. 2021-8-29
"""

import torch
import numpy as np

import learn2learn as l2l
import visdom
import os
import time

from maml_model import Net4CNN
from Datasets.cwru_data import MAML_Dataset
from train_utils import accuracy

vis = visdom.Visdom(env='yancy_meta_T1-T0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MAML_learner(object):
    def __init__(self, ways):
        h_size = 64
        layers =4
        sample_len = 1024
        feat_size = (sample_len//2**layers)*h_size
        #feat_size = 256 WDCNN时用
        self.model = Net4CNN(output_size=ways, hidden_size=h_size, layers=layers,
                             channels=1, embedding_size=feat_size).to(device)
        self.ways = ways
        # print(self.model)
    def build_tasks(self, mode='train', ways=4, shots=5, num_tasks=100, filter_labels=None):
        dataset = l2l.data.MetaDataset(MAML_Dataset(mode=mode, ways=ways))
        new_ways = len(filter_labels) if filter_labels is not None else ways
        # label_shuffle_per_task = False if ways <=30 else True
        assert shots * 2 * new_ways <= dataset.__len__()//ways*new_ways, "Reduce the number of shots!"
        tasks = l2l.data.TaskDataset(dataset, task_transforms=[
            l2l.data.transforms.FusedNWaysKShots(dataset, new_ways, 2 * shots, filter_labels=filter_labels),
            l2l.data.transforms.LoadData(dataset),
            # l2l.data.transforms.RemapLabels(dataset, shuffle=label_shuffle_per_task),
            l2l.data.transforms.RemapLabels(dataset, shuffle=True),
            # do not keep the original labels, use (0 ,..., n-1);
            # if shuffle=True, to shuffle labels at each task.
            l2l.data.transforms.ConsecutiveLabels(dataset),
            # re-order samples and make their original labels as (0 ,..., n-1).
        ], num_tasks=num_tasks)
        return tasks
    def RCNN(self, X_n):  # (5, 21, 3, 224, 224)
        N, C, W = X_n.size()
        p = np.random.rand()
        K = [1, 3, 5, 7, 11, 15]
        if p > 0.5:
            k = K[np.random.randint(0, len(K))]
            Conv = torch.nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=k//2, bias=False)
            torch.nn.init.kaiming_normal_(Conv.weight)
            X_n = Conv(X_n.reshape(-1, C, W)).reshape(N, C,  W)
        return X_n.detach()
    
    def fast_adapt1(self, data, labels):
        
        data, labels = data.to(device), labels.to(device)
        # Separate data into adaptation/evaluation sets
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)
        data=self.RCNN(data)
        return  data,labels

    def fast_adapt2(self, data, labels, learner, loss, adaptation_steps, shots, ways):
        # print('kk')
        # print(data.size(0))

        # Separate data into adaptation/evaluation sets
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)

        # Separate data into adaptation/evaluation sets
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[np.arange(shots * ways) * 2] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)  # 偶数序号为True, 奇数序号为False
        adaptation_indices = torch.from_numpy(adaptation_indices)  # 偶数序号为False, 奇数序号为True
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

        # Adapt the model
        for i in range(adaptation_steps):
            train_error = loss(learner(adaptation_data), adaptation_labels)
            learner.adapt(train_error)

        # Evaluate the adapted model
        predictions = learner(evaluation_data)
        valid_error = loss(predictions, evaluation_labels)
        valid_accuracy = accuracy(predictions, evaluation_labels)
        return valid_error, valid_accuracy

    def model_save(self, path):
        filename = path+'(1)' if os.path.exists(path) else path
        torch.save(self.model.state_dict(), filename)
        print(f'Save model at: {filename}')

    def train(self, save_path, shots=5):
        # label_shuffle_per_task=True:
        meta_lr = 0.005 # 0.005, <0.01
        fast_lr = 0.05 # 0.01

        maml = l2l.algorithms.MAML(self.model, lr=fast_lr)#对MAML原函数进行了修改，142行
        opt = torch.optim.Adam(maml.parameters(), meta_lr)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        train_ways = valid_ways = self.ways
        print(f"{train_ways}-ways, {shots}-shots for training ...")
        train_tasks = self.build_tasks('train', train_ways, shots, 1000, None)
        valid_tasks = self.build_tasks('validation', valid_ways, shots, 1000, None)
        # test_tasks = self.build_tasks('test', test_ways, shots, 1000, None)

        counter = 0
        Epochs = 2200
        meta_batch_size = 16
        adaptation_steps = 1 if shots==5 else 3

        for ep in range(Epochs):
            t0 = time.time()

            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0

            opt.zero_grad()
            for _ in range(meta_batch_size):
                # 1) Compute meta-training loss
                learner = maml.clone()
                task = train_tasks.sample()  # or a batch
                data, labels = task 
                data, labels = data.to(device), labels.to(device)
                #data,labels=self.fast_adapt1(data, labels)
                evaluation_error, evaluation_accuracy = self.fast_adapt2(data,labels, learner,loss, adaptation_steps, shots,train_ways)
    
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                # 2) Compute meta-validation loss
                learner = maml.clone()
                task = valid_tasks.sample()
                data, labels = task 
                data, labels = data.to(device), labels.to(device)
                evaluation_error, evaluation_accuracy = self.fast_adapt2(data,labels, learner, loss,
                                                                        adaptation_steps, shots, valid_ways)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            # Print some metrics
            t1 = time.time()
            print(f'Time /epoch: {t1-t0:.4f} s')
            print('\n')
            print('Iteration', ep+1)
            print(f'Meta Train Error: {meta_train_error / meta_batch_size: .4f}')
            print(f'Meta Train Accuracy: {meta_train_accuracy / meta_batch_size: .4f}')
            print(f'Meta Valid Error: {meta_valid_error / meta_batch_size: .4f}')
            print(f'Meta Valid Accuracy: {meta_valid_accuracy / meta_batch_size: .4f}')

            # Take the meta-learning step:
            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()

            vis.line(Y=[[meta_train_error / meta_batch_size, meta_valid_error / meta_batch_size]], X=[counter],
                     update=None if counter == 0 else 'append', win='Loss_MAML',
                     opts=dict(legend=['train', 'val'], title='Loss_MAML'))

            vis.line(Y=[[meta_train_accuracy / meta_batch_size, meta_valid_accuracy / meta_batch_size]], X=[counter],
                     update=None if counter == 0 else 'append', win='Acc_MAML',
                     opts=dict(legend=['train', 'val'], title='Acc_MAML'))
            counter += 1

            # if (ep + 1) >= 400 and (meta_valid_accuracy / meta_batch_size) > 0.88:
            #     new_save_path = save_path + rf'_ep{ep + 1}'
            #     self.model_save(new_save_path)
                # break
            if (ep+1) >=400 and (ep+1)%50==0:
                if input('\n== Stop training? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)
                    break
                elif input('\n== Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)

    def test(self, load_path, inner_steps=10, shots=5):
        self.model.load_state_dict(torch.load(load_path))
        print('Load Model successfully from [%s]' % load_path)

        test_ways = self.ways
        shots = shots
        print(f"{test_ways}-ways, {shots}-shots for testing ...")
        # meta_lr = 0.005  # 0.005, <0.01
        fast_lr = 0.05  # 0.01
        test_tasks = self.build_tasks('test', test_ways, shots, 1000, None)
        maml = l2l.algorithms.MAML(self.model, lr=fast_lr)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        meta_batch_size = 100
        adaptation_steps = inner_steps  # 1
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        t0 = time.time()

        for _ in range(meta_batch_size):
            # Compute meta-testing loss
            learner = maml.clone()
            task = test_tasks.sample()
            data, labels = task 
            data, labels = data.to(device), labels.to(device)
            evaluation_error, evaluation_accuracy = self.fast_adapt2(data,labels, learner, loss,
                                                                    adaptation_steps, shots, test_ways)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        t1 = time.time()
        print(f"-------- Time for {meta_batch_size*shots} samples: {t1-t0:.4f} sec. ----------")
        print(f'Meta Test Error: {meta_test_error / meta_batch_size: .4f}')
        print(f'Meta Test Accuracy: {meta_test_accuracy / meta_batch_size: .4f}\n')



#Net = MAML_learner(ways=10)  # T1
Net = MAML_learner(ways=6)  # T2
path = r"D:\\论文+无监督下基于小样本数据的旋转机械故障诊断研究\\故障诊断例子\\MetaFD-main\\Models\\MAML\\5shot_MAML_4_0_PADT0-T1"
Net.train(save_path=path, shots=5)




#test

load_path = r"D:\\论文+无监督下基于小样本数据的旋转机械故障诊断研究\\故障诊断例子\\MetaFD-main\\Models\\MAML\\5shot_MAML_4_0_PADT0-T1_ep400(1)"
Net.test(load_path, inner_steps=10, shots=1)
load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_4_0_T2-T_ep400"
Net.test(load_path, inner_steps=10, shots=2)
load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_4_0_T2-T_ep400"
Net.test(load_path, inner_steps=10, shots=1)








class Data_CWRU:
    def __init__(self, Tt=True):
        if Tt:
            self.train = T2 # T3
            self.test = T3 # T0
        else:
            self.train = T6w
            self.test = T4w  # 4 new classes

    def get_data(self, train_mode=True, n_each_class=10, sample_len=1024, normalize=True):
        data_file = self.train if train_mode else self.test
        data_size = n_each_class * sample_len
        n_way = len(data_file)  # the num of categories
        data_set = []
        for i in range(n_way):
            data = get_data_csv(file_dir=data_file[i], num=data_size, header=0, shift_step=500)  # (N, len)
            data = data.reshape(-1, sample_len)
            data = normalization(data) if normalize else data
            data_set.append(data)
        data_set = np.stack(data_set, axis=0)  # (n_way, n, sample_len)
        data_set = np.asarray(data_set, dtype=np.float32)
        label = np.arange(n_way, dtype=np.int32).reshape(n_way, 1)
        label = np.repeat(label, n_each_class, axis=1)  # [n_way, examples]
        return data_set, label  # [Nc,num_each_way,1,1024], [Nc, 50]


#Net = MAML_learner(ways=10)  # T1
Net = MAML_learner(ways=5)  # T2
path = r"D:\\论文+无监督下基于小样本数据的旋转机械故障诊断研究\\故障诊断例子\\MetaFD-main\\Models\\MAML\\5shot_MAML_4_0_SEUT3-T2"
Net.train(save_path=path, shots=5)




#test

load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_4_0_SEUT3-T2_ep300"
Net.test(load_path, inner_steps=10, shots=5)
load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_4_0_SEUT3-T2_ep300"
Net.test(load_path, inner_steps=10, shots=2)
load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_4_0_SEUT3-T2_ep300"
Net.test(load_path, inner_steps=10, shots=1)



















