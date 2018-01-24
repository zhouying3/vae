# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:48:18 2018

@author: zhouying
"""

#dataset = ['diabetes','ionosphere','satimage','segmentchallenge','vehicle','sonar']

dataset = ['vehicle']
import os,pickle
from get import get_result

for value in dataset:
    filedir = value
    if os.path.exists(value) and os.path.isfile(filedir+'\\result'):             
        with open(filedir+'.\\result','rb') as f:
            result = pickle.load(f)
        f.close()
        with open(filedir+'.\\generation1','rb') as f:
            generation = pickle.load(f)
        f.close()     
        print('###@@@======'+value+'======@@@###')
        print('###@@@=====100%=====@@@###')
        get_result(10,result,generation,value,False)
        with open(filedir+'.\\result','rb') as f:
            result = pickle.load(f)
        f.close()
        with open(filedir+'.\\generation2','rb') as f:
            generation = pickle.load(f)
        f.close()     
#        print('###@@@======'+value+'======@@@###')
        print('###@@@=====200%=====@@@###')
        get_result(10,result,generation,value,False)
        with open(filedir+'.\\result','rb') as f:
            result = pickle.load(f)
        f.close()
        with open(filedir+'.\\generation3','rb') as f:
            generation = pickle.load(f)
        f.close()     
#        print('###@@@======'+value+'======@@@###')
        print('###@@@=====300%=====@@@###')
        get_result(10,result,generation,value,False)
    else:
        print(filedir) 
        continue
