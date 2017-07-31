import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn import linear_model
import os
import json
import numpy as np
from sklearn.model_selection import KFold

#将读取的tensor进行拆分，分为训练部分和测试部分
def splitData(tensor,n_output,n_pred):
    # print(tensor.shape)
    n_known=tensor.shape[0]-n_pred
    n_input=tensor.shape[1]-n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known+n_pred, 0: n_input]
    return (knownX,knownY,preX)


#zgboost进行预测的结果
def xgb_pre(k,i):
    tensor = np.loadtxt(path + k + "\\" + i + "\\" + "tensor_fill.csv", delimiter=',')
    knownX, knownY, preX = splitData(tensor, 30, 30)
    model = xgb.XGBRegressor().fit(knownX,knownY)
    pre_y = model.predict(preX)
    return pre_y

#KNN进行预测的结果
def KNN_pre(k,i):
    tensor = np.loadtxt(path + k + "\\" + i + "\\" + "tensor_fill.csv", delimiter=',')
    knownX, knownY, preX = splitData(tensor, 30, 30)
    neigh = KNeighborsRegressor(n_neighbors=9)
    neigh.fit(knownX, knownY)
    P_x = preX.sum(axis=1).reshape(-1, 1)
    P_y = neigh.predict(P_x)
    pre_y = P_y.T[0]
    # 计算08-09各自的比例情况
    per_tensor = np.loadtxt(path + k + "\\" + i + "\\" + "tensor.csv", delimiter=',')
    rateData = per_tensor[:92, 60:].sum(axis=0).reshape(1, -1)
    rate = rateData / rateData.sum()
    r = rate[0]
    # 分摊比例，计算每两分钟的预测值
    for idx_y in pre_y:
        pre_value[i] += [idx_y * v for v in r]
    return r,pre_y

#LinearRegression预测的结果
def LinRegression(k,i):
    tensor = np.loadtxt(path + k + "\\" + i + "\\" + "tensor_fill.csv", delimiter=',')
    knownX, knownY, preX = splitData(tensor, 30, 30)
    model = linear_model.LinearRegression()
    model.fit(knownX,knownY)
    pre_y = model.predict(preX)
    return pre_y

#RandomForest预测的结果
def RandomForest(k,i):
    tensor = np.loadtxt(path + k + "\\" + i + "\\" + "tensor_fill.csv", delimiter=',')
    knownX, knownY, preX = splitData(tensor, 30, 30)
    model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2,
                                  random_state=0)
    model.fit(knownX, knownY)
    pre_y = model.predict(preX)
    return pre_y


#计算预测值与实际值之间的误差
def my_score(Y_real, Y_pred):
    MAPE = np.mean(np.abs(np.ones_like(Y_pred) - Y_pred / Y_real))
    return np.array([MAPE]).reshape(-1, 1)
#KNN与RF分别进行预测
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"
linkDict={}
with open(path+"linkDict.json") as f:
    linkDict=json.loads(f.read())
rootfiles = os.listdir(path)
pre_value={}#存储每个link的预测值
for id in range(1,133):
    pre_value[str(id)]=[]
for k in rootfiles:
    if "tensorData2" in k:
        files = os.listdir(path + k + "\\")
        for i in files:
            files_id = os.listdir(path + k + "\\" + i)
            # r:08-09每两分钟分配的比例    pre_y:KNN 预测的08-09的总值
            # r,pre_y = KNN_pre(k,i)

            pre_y = xgb_pre(k,i)

#下面的代码主要是将模型训练后预测的结果输出到最终提交文件中
timeId=[]
timeDay=[]
timeMin=[]
startDay = datetime.datetime.strptime("2016-06-01","%Y-%m-%d")
startTime = datetime.datetime.strptime("08:00:00","%H:%M:%S")
for i in range(0,30,1):
    endDay = startDay + datetime.timedelta(days=i)
    timeDay.append(datetime.datetime.strftime(endDay,"%Y-%m-%d"))
for i in range(0,62,2):
    endTime = startTime + datetime.timedelta(minutes=i)
    timeMin.append(datetime.datetime.strftime(endTime,"%H:%M:%S"))
for i in range(len(timeDay)):
    for j in range(len(timeMin)-1):
        timeId.append("["+timeDay[i]+" "+timeMin[j]+","+timeDay[i]+" "+timeMin[j+1]+")")
with open(path+"submit_XGB.txt","w") as f:
    output=[]
    for idx in pre_value:
        output += pre_value[idx]
    for i in range(len(linkDict)):
        for j in range(len(timeId)):
            f.write(linkDict[str(i+1)] + "#" + timeId[j].split(" ")[0][1:] + "#" + timeId[j] + "#" + str(output[i*900+j])+"\n")