import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os
import json
import numpy as np

def loadPath():
    with open("config.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"],config["selfvalidpath"],config["startdate"],config["days"]
datapath,sharepath,rootpath,selfvalidpath,startdate,days=loadPath()

#将读取的tensor进行拆分，分为训练部分和测试部分
def splitData(tensor,n_output,n_pred):
    # print(tensor.shape)
    n_known=tensor.shape[0]-n_pred
    n_input=tensor.shape[1]-n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known+n_pred, 0: n_input]
    return (knownX,knownY,preX)

#xgboost进行预测的结果
def xgb_pre(k,i):
    tensor = np.loadtxt(datapath + k + "\\" + i + "\\" + "tensor_all.csv", delimiter=',')
    knownX, knownY, preX = splitData(tensor, 1, 30)
    x_train, x_test, y_train, y_test = train_test_split(knownX, knownY, test_size=0.5, random_state=1)
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'count:poisson','eval_metric':'map'}
    bst = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)
    pre_data = xgb.DMatrix(preX)
    pre_y = bst.predict(pre_data)
    return pre_y

linkDict={}
with open(datapath+"linkDict.json") as f:
    linkDict=json.loads(f.read())
rootfiles = os.listdir(datapath)
pre_value={}#存储每个link的预测值
for id in range(1,133):
    pre_value[str(id)]=[]
for k in rootfiles:
    if "tensorData3" in k:
        files = os.listdir(datapath + k + "\\")
        for i in files:
            files_id = os.listdir(datapath + k + "\\" + i)
            r = []  # 08-09每两分钟分配的比例
            pre_y = []#KNN 预测的08-09的总值
            for file in files_id:
                if file == "tensor.csv":
                    with open(datapath + k + "\\" + i + "\\" + file) as f2:
                        all = f2.readlines()
                        data_dict = {}
                        for idx in range(92):
                            data_dict[idx] = []
                            values = all[idx].replace("\n", "").split(",")
                            for j in range(60, 90):
                                data_dict[idx].append(float(values[j]))
                        dat = []
                        for x in range(30):
                            dat.append(0)
                        for j in data_dict:
                            for y in range(30):
                                dat[y] += data_dict[j][y]
                        r = [x / sum(dat) for x in dat]
                    f2.closed
                elif file == "tensor_all.csv":
                    pre_y = xgb_pre(k,i)
            for idx_y in pre_y:
                pre_value[i] += [idx_y*v for v in r]
def outputResult():
    timeId=[]
    timeDay=[]
    timeMin=[]
    startDay = datetime.datetime.strptime(startdate,"%Y-%m-%d")
    startTime = datetime.datetime.strptime("08:00:00","%H:%M:%S")
    for i in range(0,int(days),1):
        endDay = startDay + datetime.timedelta(days=i)
        timeDay.append(datetime.datetime.strftime(endDay,"%Y-%m-%d"))
    for i in range(0,62,2):
        endTime = startTime + datetime.timedelta(minutes=i)
        timeMin.append(datetime.datetime.strftime(endTime,"%H:%M:%S"))
    for i in range(len(timeDay)):
        for j in range(len(timeMin)-1):
            timeId.append("["+timeDay[i]+" "+timeMin[j]+","+timeDay[i]+" "+timeMin[j+1]+")")
    with open(datapath+"submit_totalWeighted.txt","w") as f:
        output=[]
        for idx in pre_value:
            output += pre_value[idx]
        for i in range(len(linkDict)):
            for j in range(len(timeId)):
                f.write(linkDict[str(i+1)] + "#" + timeId[j].split(" ")[0][1:] + "#" + timeId[j] + "#" + str(output[i*900+j])+"\n")
if __name__ == '__main__':
    outputResult()