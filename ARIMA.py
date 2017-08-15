from statsmodels.graphics.api import qqplot
from scipy import  stats
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime
import os
import json
import SelfValidMAPE

#将读取的tensor进行拆分，分为训练部分和测试部分
def splitData(tensor,n_output,n_pred):
    n_known=tensor.shape[0]-n_pred
    n_input=tensor.shape[1]-n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known+n_pred, 0: n_input]
    return preX

def loadPath():
    # with open("config.json") as f:
    #这是用于自验证的代码
    with open("configSelfValid.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"],config["selfvalidpath"],config["startdate"],config["days"]

datapath,sharepath,rootpath,selfvalidpath,startdate,days=loadPath()



linkDict={}
with open(datapath+"linkDict.json") as f:
    linkDict=json.loads(f.read())
path = datapath
timeId=[]
timeDay=[]
timeMin=[]
startDay = datetime.datetime.strptime(startdate,"%Y-%m-%d")
startTime = datetime.datetime.strptime("08:00:00","%H:%M:%S")
for i in range(0,days,1):
    endDay = startDay + datetime.timedelta(days=i)
    timeDay.append(datetime.datetime.strftime(endDay,"%Y-%m-%d"))
for i in range(0,62,2):
    endTime = startTime + datetime.timedelta(minutes=i)
    timeMin.append(datetime.datetime.strftime(endTime,"%H:%M:%S"))
timeId=[]
for i in range(len(timeDay)):
    for j in range(len(timeMin) - 1):
        timeId.append("["+timeDay[i]+" "+timeMin[j]+","+timeDay[i]+" "+timeMin[j+1]+")")
def process():
    dataPath = selfvalidpath
    files = os.listdir(dataPath)
    pre_result={}
    for i in files:
        pre_result[i]=[]
        tensor = np.loadtxt(dataPath + i + "\\tensor_fill.csv",delimiter=',')
        preX = splitData(tensor,30,days)
        for idx in range(days):
            try:
                pre_result[i].append(ar_model(preX[idx].reshape(1,-1)[0].tolist()))
            except:
                print(i)
                value = preX[idx].reshape(1,-1)[0].tolist()[0]
                tempvalue = [value for i in range(30)]
                pre_result[i].append(tempvalue)
    return pre_result

def ar_model(data):
    dta=pd.Series(data)
    start = '1760'
    end = str(int(1760)+len(dta)-1)
    dta.index = pd.Index(sm.tsa.datetools.dates_from_range(start,end))
    arma_mod20 = sm.tsa.AR(dta).fit()
    predict_start = str(int(end)+1)
    predict_end = str(int(predict_start)+29)
    predict_sunspots = arma_mod20.predict(predict_start,predict_end,dynamic=True)
    pre_data=predict_sunspots.values.tolist()
    return pre_data

def main(output):
    result = process()
    with open(output, "w") as f:
        for k in result:
            value_list = result[k]
            preV=[]
            for i in value_list:
                preV +=i
            for j in range(len(timeId)):
                day = timeId[j].split(" ")[0][1:]
                f.write(linkDict[k] + "#" + day + "#" + timeId[j] + "#" + str(preV[j]) + "\n")
if __name__ == '__main__':
    output = "selfValid_ARIMA.txt"
    main(path + "selfValid\\"+output)
    SelfValidMAPE.processingOut(output)
    os.remove(path + "selfValid\\"+output)