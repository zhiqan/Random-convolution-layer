# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:14:58 2023

@author: Owner
"""
import numpy as np
import torch

def RCNN(X_n): 
        N, C, W = X_n.size()
        p = np.random.rand()
        K = [1, 3, 5, 7, 11, 15]
        if p > 0.5:
            k = K[np.random.randint(0, len(K))]
            Conv = torch.nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=k//2, bias=False)
            torch.nn.init.kaiming_normal_(Conv.weight)
            X_n = Conv(X_n.reshape(-1, C, W)).reshape(N, C,  W)
        return X_n.detach()
