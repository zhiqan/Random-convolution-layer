"""
Code refers to: https://github.com/learnables/learn2learn/blob/master/examples/vision/maml_omniglot.py
By Yancy F. 2021-8-29
"""

import torch
import numpy as np
# from torch.utils.data import DataLoader

import learn2learn as l2l
import visdom
import os
import time

from Models.ProtoNet.proto_model import Net4CNN
from Datasets.cwru_data import MAML_Dataset
from my_utils.train_utils import accuracy
from my_utils.init_utils import weights_init2

     
import torchextractor as tx

vis = visdom.Visdom(env='yancy_meta')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProtoNet_learner(object):
    def __init__(self, ways):
        super().__init__()
        self.model = Net4CNN(hidden_size=64, layers=3, channels=1).to(device)
        self.ways = ways
    def RCNN(self, X_n):  # (5, 21, 3, 224, 224)
        N, C, W = X_n.size()
        p = np.random.rand()
        K = [1, 3, 5, 7, 11, 15]
        if p > 0.5:
            k = K[np.random.randint(0, len(K))]
            Conv = torch.nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=k//2, bias=False)
            torch.nn.init.xavier_normal_(Conv.weight)
            X_n = Conv(X_n.reshape(-1, C, W)).reshape(N, C,  W)
        return X_n.detach()
    
    def fast_adapt1(self, data, labels,query_num, shots, ways):
        
        data, labels = data.to(device), labels.to(device)
        # Separate data into adaptation/evaluation sets
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)
        data=self.RCNN(data)
        return  data,labels


    @staticmethod
    def build_tasks(mode='train', ways=10, shots=5, num_tasks=100, filter_labels=None):
        dataset = l2l.data.MetaDataset(MAML_Dataset(mode=mode, ways=ways))
        new_ways = len(filter_labels) if filter_labels is not None else ways
        # label_shuffle_per_task = False if ways <=30 else True
        assert shots * 2 * new_ways <= dataset.__len__() // ways * new_ways, "Reduce the number of shots!"
        tasks = l2l.data.TaskDataset(dataset, task_transforms=[
            l2l.data.transforms.FusedNWaysKShots(dataset, new_ways, 2 * shots, filter_labels=filter_labels),
            l2l.data.transforms.LoadData(dataset),
            # l2l.data.transforms.RemapLabels(dataset, shuffle=label_shuffle_per_task),
            l2l.data.transforms.RemapLabels(dataset, shuffle=True),
            # do not keep the original labels, use (0 ,..., n-1);
            # if shuffle=True, to shuffle labels at each task,
            # do not recommend this when number of ways is little (<30)
            l2l.data.transforms.ConsecutiveLabels(dataset),
            # re-order samples and make their original labels as (0 ,..., n-1).
        ], num_tasks=num_tasks)
        return tasks

    @staticmethod
    def euclidean_metric(query_x, proto_x):
        """
        :param query_x: (n, d), N-example, P-dimension for each example; zq
        :param proto_x: (nc, d), M-Way, P-dimension for each example, but 1 example for each Way; z_proto
        :return: [n, nc]
        """
        query_x = query_x.unsqueeze(dim=1)  # [n, d]==>[n, 1, d]
        proto_x = proto_x.unsqueeze(dim=0)  # [nc, d]==>[1, nc, d]
        logits = -torch.pow(query_x - proto_x, 2).mean(dim=2)  # (n, nc)
        return logits

    def fast_adapt(self, data, labels, learner, loss_fun, query_num, shots, ways):
        # print('kk')
        # print(data.size(0))

        # Separate data into adaptation/evaluation sets
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)

        support_indices = np.zeros(data.size(0), dtype=bool)
        # print(data.size(0))
        selection = np.arange(ways) * (shots + query_num)  # 0, shot+q, 2*(shot+q), 3*(), ...
        for offset in range(shots):
            support_indices[selection + offset] = True  # 0:shots, (shot+q):(shot+q+shots), ...

        query_indices = torch.from_numpy(~support_indices)  # shots:2*shots, (shot+q+shots):4*shots, ...
        support_indices = torch.from_numpy(support_indices)  # 0:shots, (shot+q):(shot+q+shots), ...

        
        model = tx.Extractor(learner, ["features2"])
        embeddings1,feature = model(data)
        embeddings = learner(data)
        
        support = embeddings[support_indices]
        support = support.reshape(ways, shots, -1).mean(dim=1)  # (ways, dim)
        query = embeddings[query_indices]  # (n_query, dim)
        labels = labels[query_indices].long()

        logits = self.euclidean_metric(query, support)
        error = loss_fun(logits, labels)
        acc = accuracy(logits, labels)
        return error, acc,feature,logits,labels

    def model_save(self, path):
        filename = path+'(1)' if os.path.exists(path) else path
        torch.save(self.model.state_dict(), filename)
        print(f'Save model at: {filename}')

    def train(self, save_path, shots):
        train_ways = valid_ways = self.ways
        query_num = shots
        print(f"{train_ways}-ways, {shots}-shots for training ...")
        train_tasks = self.build_tasks('train', train_ways, shots, 1000, None)
        valid_tasks = self.build_tasks('validation', valid_ways, shots, 50, None)
        # valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

        self.model.apply(weights_init2)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        loss_fun = torch.nn.CrossEntropyLoss()

        Epochs = 10000
        Episodes = 30
        counter = 0
        out_output_train=[]
        out_label_train=[]
        out=[]
        data1=[]
        out_label=[]

        for ep in range(Epochs):
            # 1) training:
            t0 = time.time()
            self.model.train()    
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            out_label_test=[]
            for epi in range(Episodes):
                batch = train_tasks.sample()
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
                data,labels=self.fast_adapt1(data, labels,query_num, shots, train_ways)
                
                loss, acc,feature ,logits,label = self.fast_adapt(data,labels, self.model, loss_fun, query_num, shots, train_ways)
                meta_train_error += loss.item()
                meta_train_accuracy += acc.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            t1 = time.time()
            data1.append(data)
            out_label_train.append(labels[:, np.newaxis])
            out_label.append(label[:, np.newaxis])
           
            feature=feature['features2'].detach().numpy()
            out.append(logits.detach().numpy())
            out_output_train.append(feature)
            print(f'*** Time /epoch {t1-t0:.4f} ***')
            print(f'epoch {ep+1}, train, loss: {meta_train_error/Episodes:.4f}, '
                  f'acc: {meta_train_accuracy/Episodes:.4f}')

            # 2) validation:
            self.model.eval()
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0
            for i, batch in enumerate(valid_tasks):
                with torch.no_grad():
                    data, labels = batch
                    data, labels = data.to(device), labels.to(device)
                    loss, acc,_,_ ,_= self.fast_adapt(data,labels, self.model, loss_fun, query_num, shots, train_ways)
                meta_valid_error += loss.item()
                meta_valid_accuracy += acc.item()
            
            print(f'epoch {ep + 1}, validation, loss: {meta_valid_error / len(valid_tasks):.4f}, '
                  f'acc: {meta_valid_accuracy / len(valid_tasks):.4f}\n')

            vis.line(Y=[[meta_train_error / Episodes, meta_valid_error / len(valid_tasks)]], X=[counter],
                     update=None if counter == 0 else 'append', win='Loss_ProtoNet',
                     opts=dict(legend=['train', 'val'], title='Loss_ProtoNet'))

            vis.line(Y=[[meta_train_accuracy / Episodes, meta_valid_accuracy / len(valid_tasks)]], X=[counter],
                     update=None if counter == 0 else 'append', win='Acc_ProtoNet',
                     opts=dict(legend=['train', 'val'], title='Acc_ProtoNet'))
            counter += 1

            if (ep+1) >=80 and (ep+1)%2==0:
                if input('\n== Stop training? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    b  = np.concatenate(out_output_train, axis=0)
                    target_array_train = np.concatenate(out_label_train, axis=0)
                    out2=np.concatenate(out, axis=0)
                    data2=np.concatenate(data1, axis=0)
                    out_label1=np.concatenate(out_label, axis=0)
                    np.save(os.path.join('D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML', 'feature.npy'),  b, allow_pickle=False)
                    np.save(os.path.join('D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML', 'feature_label.npy'),  target_array_train , allow_pickle=False)
                    np.save(os.path.join('D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML', 'out.npy'),  out2 , allow_pickle=False)
                    np.save(os.path.join('D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML', 'data.npy'),  data2 , allow_pickle=False)
                    np.save(os.path.join('D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML', ' out_label.npy'),   out_label1 , allow_pickle=False)
                    self.model_save(new_save_path)
                    break
                elif input('\n== Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)

    def test(self, load_path, shots):
        self.model.load_state_dict(torch.load(load_path))
        print(f'Load Model successfully from [{load_path}]...')
        out_label_test=[]
        out_output_test=[]

        test_ways = self.ways
        query_num = shots
        print(f"{test_ways}-ways, {shots}-shots for testing ...")
        test_tasks = self.build_tasks('test', test_ways, shots, 1000, None)
        loss_fun = torch.nn.CrossEntropyLoss()

        # 2) validation:
        self.model.eval()
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        t0 = time.time()
        for i, batch in enumerate(test_tasks):
            with torch.no_grad():
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
                loss, acc,feature,_ ,_= self.fast_adapt(data,labels, self.model, loss_fun, query_num, shots, test_ways)
                out_label_test.append(labels[:, np.newaxis])
                feature= feature['features2'].detach().numpy()
                out_output_test.append(feature)
                
            meta_valid_error += loss.item()
            meta_valid_accuracy += acc.item()

        #model_layer= list(self.model.children())
        #model_layer=model_layer[0]#这里选择model的第一个Sequential()
        b  = np.concatenate(out_output_test, axis=0)
        target_array_test = np.concatenate(out_label_test, axis=0)
        np.save(os.path.join('D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML', 'feature_test.npy'),  b, allow_pickle=False)
        np.save(os.path.join('D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-main\Models\MAML', 'label_test.npy'),  target_array_test , allow_pickle=False)
        
        t1 = time.time()
        print(f"*** Time for {len(test_tasks)} tasks: {t1-t0:.4f} (s)")
        print(f'Testing, loss: {meta_valid_error / len(test_tasks):.4f}, '
              f'acc: {meta_valid_accuracy / len(test_tasks):.4f}')


if __name__ == "__main__":
    from my_utils.init_utils import seed_torch

    seed_torch(2021)
    
    #Net = ProtoNet_learner(ways=4)  # T2

    if input('Train? y/n\n').lower() == 'y':
        # path = r"G:\model_save\meta_learning\ProtoNet\5shot\ProtoNet_C30"
        # Net.train(save_path=path, shots=5)

        # path = r"G:\model_save\meta_learning\ProtoNet\1shot\ProtoNet_C30"
        # Net.train(save_path=path, shots=1)

        # path = r"G:\model_save\meta_learning\ProtoNet\5shot\ProtoNet_T2"
        # Net.train(save_path=path, shots=5)
        Net = ProtoNet_learner(ways=3)  # T1
        path = r"D:\\论文+无监督下基于小样本数据的旋转机械故障诊断研究\\故障诊断例子\\MetaFD-main\\Models\\ProtoNet\\5shot_ProtoNet_T2-T3"
        Net.train(save_path=path, shots=5)

    if input('Test? y/n\n').lower() == 'y':
        # load_path = r"G:\model_save\meta_learning\ProtoNet\5shot\ProtoNet_C30_ep50"  # acc: 0.953
        # # load_path = r"G:\model_save\meta_learning\ProtoNet\5shot\ProtoNet_C30_ep52"  # acc: 0.950
        # Net.test(load_path, shots=5)

        # load_path = r"G:\model_save\meta_learning\ProtoNet\1shot\ProtoNet_C30_ep102"  # acc: 0.953
        # Net.test(load_path, shots=1)

        # load_path = r"G:\model_save\meta_learning\ProtoNet\5shot\ProtoNet_T2_ep62"
        # Net.test(load_path, shots=5)

        load_path =  r"D:\\论文+无监督下基于小样本数据的旋转机械故障诊断研究\\故障诊断例子\\MetaFD-main\\Models\\ProtoNet\\5shot_ProtoNet_T2-T3_ep50"
        Net.test(load_path, shots=5)
