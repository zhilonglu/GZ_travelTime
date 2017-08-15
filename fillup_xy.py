import os
import json
import numpy as np
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\tensorData5\\"

#将tensor_fill文件填补成tensor_fill2文件
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

files = os.listdir(path)
def fillupFirst():
    #填补成tensor_fill文件
    for i in files:
        with open(path+i+"\\tensor_fill.csv","w") as f1:
            with open(path+i+"\\tensor.csv") as f2:
                all = f2.readlines()
                for j in range(len(all)):
                    str_list = list(map(float,all[j].replace("\n","").split(",")))
                    for k in range(1,len(str_list)-1):
                        if str_list[k]==0:
                            if 0 in [str_list[k - 1], str_list[k + 1]]:#若有0取最大的
                                str_list[k] = max(str_list[k - 1], str_list[k + 1])
                            else:#若没0取最小的
                                str_list[k] = min(str_list[k - 1], str_list[k + 1])  # 相邻两个元素取最小
                    if str_list[len(str_list)-1]==0:
                        str_list[len(str_list) - 1] = str_list[len(str_list) - 2]
                    for m in range(len(str_list)-2,0,-1):
                        if str_list[m]==0:
                            if 0 in [str_list[m - 1], str_list[m + 1]]:
                                str_list[m] = max(str_list[m - 1], str_list[m + 1])
                            else:
                                str_list[m] = min(str_list[m - 1], str_list[m + 1])  # 相邻两个元素取最小
                    if str_list[0]==0:
                        str_list[0] = str_list[1]
                    f1.write(",".join(list(map(str,str_list)))+"\n")
def fillupSecond():
    for i in files:
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
        for row in range(tensor_fill.shape[0]):
            for col in range(tensor_fill.shape[1]):
                if tensor_fill[row][col] ==0:
                    tensor_fill[row][col] = global_min
        os.remove(path + i + "\\tensor_fill.csv")
        np.savetxt(path+i+"\\tensor_fill.csv",tensor_fill,fmt="%.4f",delimiter=',')
if __name__ == '__main__':
    fillupFirst()
    fillupSecond()