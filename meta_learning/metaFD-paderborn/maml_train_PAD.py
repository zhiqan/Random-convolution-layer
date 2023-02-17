"""
Code refers to: https://github.com/learnables/learn2learn/blob/master/examples/vision/maml_omniglot.py
By Yancy F. 2021-8-29
"""

import torch
import torch.nn.functional as F
import numpy as np
import math

import learn2learn as l2l
import visdom
import os
import time

#from maml_model import Net4CNN
#from Datasets.cwru_data import MAML_Dataset
#from train_utils import accuracy

vis = visdom.Visdom(env='7_12_AIML_T0-T1-tgd2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MAML_learner(object):
    def __init__(self, ways):
        h_size = 64
        layers = 7
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

    @staticmethod
    def fast_adapt(data,labels, learner, loss, adaptation_steps, shots, ways):
       
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
            aa=learner.adapt(train_error)


        # Evaluate the adapted model
        predictions = learner(evaluation_data)
        valid_error = loss(predictions, evaluation_labels)
        valid_accuracy = accuracy(predictions, evaluation_labels)
        return valid_error, valid_accuracy,aa

    def model_save(self, path):
        filename = path+'(1)' if os.path.exists(path) else path
        torch.save(self.model.state_dict(), filename)
        print(f'Save model at: {filename}')

    def train(self, save_path, shots=5):
        # label_shuffle_per_task=True:
        meta_lr = 0.005 # 0.005, <0.01
        fast_lr = 0.05 # 0.01

        maml = l2l.algorithms.MAML(self.model, lr=fast_lr,allow_nograd=True)#对MAML原函数进行了修改，142行      
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
        t_g=[]
        t_g1=[]

        for ep in range(1000):
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

                
                N, C, W = data.size()
                p = np.random.rand()
                K = [1, 3, 5, 7, 11, 15]
                if p > 0.5:
                    k = K[np.random.randint(0, len(K))]
                    Conv = torch.nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=k//2, bias=False)
                    torch.nn.init.xavier_normal_(Conv.weight)
                    data = Conv(data.reshape(-1, C, W)).reshape(N, C,  W)
                evaluation_error, evaluation_accuracy,aa = self.fast_adapt(data,labels, learner, loss,
                                                                   adaptation_steps, shots, train_ways)   
                

                
                
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                

                # 2) Compute meta-validation loss
                learner = maml.clone()
                task = valid_tasks.sample()
                data, labels = task
                data, labels = data.to(device), labels.to(device)

                evaluation_error, evaluation_accuracy,ab = self.fast_adapt(data,labels, learner, loss,
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
            #for p in list(maml.parameters())[12:]:
                #p.grad.data.mul_(1.0 / meta_batch_size)
            #opt.step()
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()
            bb=list(maml.parameters())[25].grad.data
            bb1=bb.tolist()
            t_g.append(bb1)
            if counter>1:
                cc=math.acos(np.dot(t_g[counter],t_g[counter-1])/np.linalg.norm(t_g[counter])/np.linalg.norm(t_g[counter-1]))
                vis.line(Y=[cc], X=[counter],update=None if counter == 0 else 'append', win='degree of tasks1')
            
            #cc1=torch.acos_(ab[0].dot(bb)/ torch.norm(ab[0])/ torch.norm(ab[0]))
            
            aa1=aa[25]
            aa1=aa1.tolist()
            t_g1.append(aa1)
            if counter>1:
                cc=math.acos(np.dot(t_g1[counter],t_g1[counter-1])/np.linalg.norm(t_g1[counter])/np.linalg.norm(t_g1[counter-1]))
                vis.line(Y=[cc], X=[counter],update=None if counter == 0 else 'append', win='degree of tasks2')
            #vis.line(Y=[cc1.tolist()], X=[counter],update=None if counter == 0 else 'append', win='degree of tasks2')

            

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
            if (ep+1) >=200 and (ep+1)%20==0:
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
        maml = l2l.algorithms.MAML(self.model, lr=fast_lr,allow_nograd=True)#,allow_nograd=True
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        meta_batch_size = 100
        adaptation_steps = inner_steps  # 1
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        t0 = time.time()
        t_g1=[]
        for counter in range(meta_batch_size):
            # Compute meta-testing loss
            learner = maml.clone()
            task = test_tasks.sample()
            data, labels = task
            data, labels = data.to(device), labels.to(device)
            evaluation_error, evaluation_accuracy,aa= self.fast_adapt(data,labels, learner, loss,
                                                                    adaptation_steps, shots, test_ways)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()
            aa1=aa[25]
            aa1=aa1.tolist()
            t_g1.append(aa1)
            if counter>1:
                cc=math.acos(np.dot(t_g1[counter],t_g1[counter-1])/np.linalg.norm(t_g1[counter])/np.linalg.norm(t_g1[counter-1]))
                vis.line(Y=[cc], X=[counter],update=None if counter == 0 else 'append', win='degree of tasks2')

        t1 = time.time()
        print(f"-------- Time for {meta_batch_size*shots} samples: {t1-t0:.4f} sec. ----------")
        print(f'Meta Test Error: {meta_test_error / meta_batch_size: .4f}')
        print(f'Meta Test Accuracy: {meta_test_accuracy / meta_batch_size: .4f}\n')



#from my_utils.init_utils import seed_torch

#seed_torch(2021)

#Net = MAML_learner(ways=10)  # T1
#Net = MAML_learner(ways=4)  # T2
#Net = MAML_learner(ways=5) #seu_dataset
Net = MAML_learner(ways=3) #PU_dataset

if input('Train? y/n\n').lower() == 'y':
    # path = r"G:\model_save\meta_learning\MAML\5shot\MAML_C30"
    # Net.train(save_path=path, shots=5)

    # path = r"G:\model_save\meta_learning\MAML\1shot\MAML_C30"
    # Net.train(save_path=path, shots=1)

    # path = r"G:\model_save\meta_learning\MAML\5shot\MAML_T2"
    # Net.train(save_path=path, shots=5)

    path = r"D:\\论文+无监督下基于小样本数据的旋转机械故障诊断研究\\故障诊断例子\\MetaFD-main\\Models\\MAML\\5shot_MAML_7_n12_SEUT0-T1==t1"
    Net.train(save_path=path, shots=5)

if input('Test? y/n\n').lower() == 'y':
    # load_path = r"G:\model_save\meta_learning\MAML\5shot\MAML_C30_ep457"  # acc: 0.847; 0.958
    # Net.test(load_path, inner_steps=10, shots=5)

    # load_path = r"G:\model_save\meta_learning\MAML\1shot\MAML_C30_ep436"  # acc: 0.874
    # Net.test(load_path, inner_steps=20, shots=1)

    # load_path = r"G:\model_save\meta_learning\MAML\5shot\MAML_T2_ep404"  # acc:
    # Net.test(load_path, inner_steps=10, shots=5)

    load_path = r"D:\\论文+无监督下基于小样本数据的旋转机械故障诊断研究\\故障诊断例子\\MetaFD-main\\Models\\MAML\\5shot_MAML_7_n12_SEUT0-T1==t1_ep200(1)"
    Net.test(load_path, inner_steps=10, shots=5)
    load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_7_12_PAD1T1-T0_ep260"
    Net.test(load_path, inner_steps=10, shots=2)
    load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_7_12_PAD1T1-T0_ep260"
    Net.test(load_path, inner_steps=10, shots=1)












class Data_CWRU:
    def __init__(self, Tt=True):
        if Tt:
            self.train = T1 # T3
            self.test = T0 # T0
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





path = r"D:\\论文+无监督下基于小样本数据的旋转机械故障诊断研究\\故障诊断例子\\MetaFD-main\\Models\\MAML\\5shot_MAML_7_12_PADT3-T0"
Net.train(save_path=path, shots=5)

load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_7_12_PADT3-T0_ep260"
Net.test(load_path, inner_steps=10, shots=5)
load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_7_12_PADT3-T0_ep260"
Net.test(load_path, inner_steps=10, shots=2)
load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_7_12_PADT3-T0_ep260"
Net.test(load_path, inner_steps=10, shots=1)



class MAML_learner(object):
    def __init__(self, ways):
        h_size = 64
        layers = 7
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

    @staticmethod
    def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)

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

        maml = l2l.algorithms.MAML(self.model, lr=fast_lr)#对MAML原函数进行了修改，142行,allow_nograd=True
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
                evaluation_error, evaluation_accuracy = self.fast_adapt(task, learner, loss,
                                                                   adaptation_steps, shots, train_ways)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                # 2) Compute meta-validation loss
                learner = maml.clone()
                task = valid_tasks.sample()
                evaluation_error, evaluation_accuracy = self.fast_adapt(task, learner, loss,
                                                                        adaptation_steps, shots, valid_ways)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            # Print some metrics
            #t1 = time.time()
            #print(f'Time /epoch: {t1-t0:.4f} s')
            #print('\n')
            #print('Iteration', ep+1)
            #print(f'Meta Train Error: {meta_train_error / meta_batch_size: .4f}')
            #print(f'Meta Train Accuracy: {meta_train_accuracy / meta_batch_size: .4f}')
            #print(f'Meta Valid Error: {meta_valid_error / meta_batch_size: .4f}')
            #print(f'Meta Valid Accuracy: {meta_valid_accuracy / meta_batch_size: .4f}')

            # Take the meta-learning step:
            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()

            #vis.line(Y=[[meta_train_error / meta_batch_size, meta_valid_error / meta_batch_size]], X=[counter],
                     #update=None if counter == 0 else 'append', win='Loss_MAML',
                     #opts=dict(legend=['train', 'val'], title='Loss_MAML'))

            #vis.line(Y=[[meta_train_accuracy / meta_batch_size, meta_valid_accuracy / meta_batch_size]], X=[counter],
                     #update=None if counter == 0 else 'append', win='Acc_MAML',
                     #opts=dict(legend=['train', 'val'], title='Acc_MAML'))
            #counter += 1

            # if (ep + 1) >= 400 and (meta_valid_accuracy / meta_batch_size) > 0.88:
            #     new_save_path = save_path + rf'_ep{ep + 1}'
            #     self.model_save(new_save_path)
                # break
            if (ep+1) >=260 and (ep+1)%20==0:
                #if input('\n== Stop training? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)
                    break
                #elif input('\n== Save model? == (y/n)\n').lower() == 'y':
                    #new_save_path = save_path + rf'_ep{ep + 1}'
                    #self.model_save(new_save_path)

    def test(self, load_path, inner_steps=10, shots=5):
        self.model.load_state_dict(torch.load(load_path))
        print('Load Model successfully from [%s]' % load_path)

        test_ways = self.ways
        shots = shots
        print(f"{test_ways}-ways, {shots}-shots for testing ...")
        # meta_lr = 0.005  # 0.005, <0.01
        fast_lr = 0.05  # 0.01
        test_tasks = self.build_tasks('test', test_ways, shots, 1000, None)
        maml = l2l.algorithms.MAML(self.model, lr=fast_lr,allow_nograd=True)
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
            evaluation_error, evaluation_accuracy = self.fast_adapt(task, learner, loss,
                                                                    adaptation_steps, shots, test_ways)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        t1 = time.time()
        print(f"-------- Time for {meta_batch_size*shots} samples: {t1-t0:.4f} sec. ----------")
        print(f'Meta Test Error: {meta_test_error / meta_batch_size: .4f}')
        print(f'Meta Test Accuracy: {meta_test_accuracy / meta_batch_size: .4f}\n')







class Data_CWRU:
    def __init__(self, Tt=True):
        if Tt:
            self.train = T1 # T3
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



Net = MAML_learner(ways=3) #seu_dataset

path = r"D:\\论文+无监督下基于小样本数据的旋转机械故障诊断研究\\故障诊断例子\\MetaFD-main\\Models\\MAML\\5shot_MAML_7_12_PADT1-T3"
Net.train(save_path=path, shots=5)

load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_7_12_PADT1-T3_ep260"
Net.test(load_path, inner_steps=10, shots=5)
load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_7_12_PADT1-T3_ep260"
Net.test(load_path, inner_steps=10, shots=2)
load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_7_12_PADT1-T3_ep260"
Net.test(load_path, inner_steps=10, shots=1)






class Data_CWRU:
    def __init__(self, Tt=True):
        if Tt:
            self.train = T3 # T3
            self.test = T1 # T0
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


Net = MAML_learner(ways=3) #seu_dataset


path = r"D:\\论文+无监督下基于小样本数据的旋转机械故障诊断研究\\故障诊断例子\\MetaFD-main\\Models\\MAML\\5shot_MAML_7_12_PADT3-T1"
Net.train(save_path=path, shots=5)

load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_7_12_PADT3-T1_ep260"
Net.test(load_path, inner_steps=10, shots=5)
load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_7_12_PADT3-T1_ep260"
Net.test(load_path, inner_steps=10, shots=2)
load_path = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML\5shot_MAML_7_12_PADT3-T1_ep260"
Net.test(load_path, inner_steps=10, shots=1)


