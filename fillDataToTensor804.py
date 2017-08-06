import datetime
import os
import json
import numpy as np
rootpath = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"
path = rootpath+"tensorData4\\"


def splitData(tensor,n_output):
    n_input=tensor.shape[1]-n_output
    knownX = tensor[:, 0: n_input]
    knownY = tensor[:, n_input: n_input + n_output]
    return (knownX,knownY)
link_top = {}
for i in range(1,133):
    link_top[str(i)] = []
with open(rootpath+"new_top.txt") as f:
    f.readline()  # skip the header
    all = f.readlines()
    for line in all:
        values = line.replace("\n","").replace("#",";").split(";")
        for i in range(1,len(values)):
            if values[i]!="": #若为空则不加入数组
                link_top[values[0]].append(values[i])
files = os.listdir(path)
for i in files:
    files_id = os.listdir(path+i)
    links = link_top[i]
    tensor = np.loadtxt(path+i+ "\\tensor_fill.csv", delimiter=',')
    trainX,trainY = splitData(tensor,30)
    link_tensor = {}
    link_trainX = {}
    link_trainY = {}
    for col in range(trainX.shape[1]):
        if col == 0:
            output_tensor = trainX[:,col]
        else:
            output_tensor = np.c_[output_tensor,trainX[:, col]]
        for j in links:
            link_tensor[j] = np.loadtxt(path+j+ "\\tensor_fill.csv", delimiter=',')
            link_trainX[j],link_trainY[j] = splitData(link_tensor[j],30)
            output_tensor = np.c_[output_tensor,link_trainX[j][:,col]]
    output_tensor = np.c_[output_tensor,trainY]
    np.savetxt(path+i+"\\"+"tensor_extend.csv",output_tensor, fmt="%.5f", delimiter=',')