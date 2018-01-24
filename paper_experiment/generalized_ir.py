# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:31:40 2018

@author: zhouying
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.io as scio

mydata = scio.loadmat('MNIST_data\\UCI\\sonar.mat')
data = np.array(mydata['data'])
label = np.transpose(mydata['label'])
label = label[:,0]

nbrs = NearestNeighbors(n_neighbors=6, algorithm="auto").fit(data)
_,indices = nbrs.kneighbors(data)

positvie =1
negative = 2

nn = indices[:,1:]
Tplas = 0.
Tminas=0.
for i in range(label.shape[0]):
    if label[i] == positvie :
        for value in label[nn[i]]:
            if value == positvie:
                Tplas+=1
    elif label[i]==negative:
        for value in label[nn[i]]:
            if value == negative:
                Tminas+=1
        
        
Tplas/=label[label==positvie].shape[0]*nn.shape[1]
Tminas/=label[label==negative].shape[0]*nn.shape[1]

print(Tplas,Tminas,Tminas-Tplas)
