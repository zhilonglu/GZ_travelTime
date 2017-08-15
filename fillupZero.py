import os
import numpy as np
import json
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\tensorData5\\"


def splitData(tensor,n_output):
    n_input=tensor.shape[1]-n_output
    knownX = tensor[:, 0: n_input]
    knownY = tensor[:, n_input: n_input + n_output]
    return (knownX,knownY)

#找出数组中最小的非0元素
def findMinExceptZero(a):
    tmp = sorted(a)
    for i in tmp:
        if i != 0:
            return i
    return 0

datapath = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"
linkDict={}
with open(datapath+"linkDict.json") as f:
    linkDict=json.loads(f.read())

files = os.listdir(path)
lst_error = ["36","38","67","68"]
for i in files:
    if i in lst_error:
        global_min = 0
        isFirst = True
        tensor_fill = np.loadtxt(path+i+ "\\tensor_fill.csv", delimiter=',')
        for j in range(tensor_fill.shape[0]):
            temp = findMinExceptZero(tensor_fill[j,:])
            if isFirst and temp !=0:
                global_min = temp
                isFirst = False
            elif temp<global_min and temp !=0:
                global_min = temp
        print(linkDict[i]+"_min:"+str(global_min))