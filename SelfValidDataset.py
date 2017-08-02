import json
import numpy as np
import os

def listmap(o,p):
    return list(map(o,p))

def loadPath():
    with open("config.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"]

datapath,sharepath,rootpath=loadPath()

outpath=datapath+"tensorData4\\"
if not os.path.exists(outpath):
    os.makedirs(outpath)

ids = listmap(str, list(range(1,133)))

for i in ids:
    taskpath=rootpath+i+"\\"
    taskoutpath=outpath+i+"\\"
    if not os.path.exists(taskoutpath):
        os.makedirs(taskoutpath)
    tensor=np.loadtxt(taskpath+"tensor.csv",delimiter=',')
    tensor_fill=np.loadtxt(taskpath+"tensor_fill.csv",delimiter=',')
    tensor=tensor[0:-30,:]
    tensor_fill=tensor_fill[0:-30,:]
    np.savetxt(taskoutpath + "tensor.csv", tensor, fmt="%.8f", delimiter=',')
    np.savetxt(taskoutpath + "tensor_fill.csv", tensor_fill, fmt="%.8f", delimiter=',')

