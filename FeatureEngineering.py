import datetime
from sklearn import gaussian_process
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor,AdaBoostRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
import os
import json
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import SelfValidMAPE
from sklearn.preprocessing import StandardScaler

def loadPath():
    ##这是用于自验证的代码
    with open("configSelfValid.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"],config["selfvalidpath"],config["startdate"],config["days"]
datapath,sharepath,rootpath,selfvalidpath,startdate,days=loadPath()



def get_model_list():
    model_list, name_list = [], []

    # model_list.append(linear_model.LinearRegression())
    # name_list.append('LR')

    # model_list.append(gaussian_process.GaussianProcessRegressor(alpha=1e-10))
    # name_list.append('GaussianProcess')

    # model_list.append(KNeighborsRegressor(weights = 'uniform',n_neighbors=28))
    # name_list.append('KNN_unif')
    #
    # model_list.append(KNeighborsRegressor(weights = 'distance',n_neighbors=28))
    # name_list.append('KNN_dist')
    #
    # model_list.append(SVR(kernel = 'poly', C = 1, gamma = 'auto', coef0 = 0, degree = 2))
    # name_list.append('SVR_poly')
    # #
    model_list.append(SVR(kernel = 'rbf', C = 0.3, gamma = 'auto'))
    name_list.append('SVR_rbf')
    # #
    # model_list.append(DecisionTreeRegressor())
    # name_list.append('DT')
    #
    # model_list.append(RandomForestRegressor(n_estimators=150, max_depth=None,min_samples_split=2, random_state=0))
    # name_list.append('RF')
    #
    # model_list.append(ExtraTreesRegressor(n_estimators=150, max_depth=None, max_features='auto', min_samples_split=2, random_state=0))
    # name_list.append('ET')

    return model_list,name_list

#特征提取
def feature_extraction(X):
    # time series feature
    X_self = X.copy()
    # statical feature
    X_p0, X_p25, X_p50, X_p75, X_p100 = np.percentile(X, (0, 25, 50, 75, 100),axis = 1)
    X_mean = np.mean(X, axis = 1)
    X_sum = np.sum(X, axis = 1)
    X_std = np.std(X, axis = 1)
    X_var = np.var(X, axis = 1)
    X_diff = np.diff(X, axis = 1)
    X_diff2 = np.diff(X_diff, axis = 1)
    X_statical = np.c_[X_p0,X_p25,X_p50,X_p100,X_mean, X_sum, X_std, X_var,X_diff,X_diff2]

    # discrete feature
    X_int01 = (X / 1).astype(np.int)
    X_int05 = (X / 5).astype(np.int)
    X_int10 = (X / 10).astype(np.int)
    X_int20 = (X / 20).astype(np.int)
    X_int30 = (X / 30).astype(np.int)
    X_int40 = (X / 40).astype(np.int)
    X_int50 = (X / 50).astype(np.int)
    X_discrete = np.c_[X_int01,X_int05,X_int10]

    if True:
        return np.c_[X_self,X_statical,X_discrete]

#目标函数
def mapeobj(preds,dtrain):
    # gaps = dtrain.get_label()
    gaps = dtrain
    grad = np.sign(preds-gaps)/gaps
    hess = 1/gaps
    grad[(gaps==0)] = 0
    hess[(gaps==0)] = 0
    return grad,hess

#评估函数
def mape(preds,dtrain):
    # Y_real = dtrain.get_label()
    Y_real = dtrain
    Y_pred = preds
    loss = 0
    cnt = 0
    for i in range(len(Y_real)):
        if float(Y_real[i]) == 0:
            continue
        else:
            loss += abs(float(Y_pred[i])/float(Y_real[i])-1)
            cnt += 1
    return "mape",loss/cnt

def my_score(Y_real, Y_pred):
    MAPE = np.mean(np.abs(np.ones_like(Y_pred) - Y_pred / Y_real))
    return np.array([MAPE]).reshape(-1, 1)

#将读取的tensor进行拆分，分为训练部分和测试部分
def splitData(tensor,n_output,n_pred):
    n_known=tensor.shape[0]-n_pred
    n_input=tensor.shape[1]-n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known+n_pred, 0: n_input]
    return (knownX,knownY,preX)

#xgboost进行预测的结果
def xgb_Train(knownX,knownY,preX,tempPara):
    x_train, x_test, y_train, y_test = train_test_split(knownX, knownY, test_size=0.5, random_state=1)
    for i in range(y_train.shape[1]):
        data_train = xgb.DMatrix(x_train, label=y_train[:, i].reshape(-1, 1))# 按列训练，分30次训练
        param = { 'n_estimators': 1000, 'max_depth': tempPara[1],
                 'min_child_weight': 5, 'gamma': 0, 'subsample': tempPara[2], 'colsample_bytree': tempPara[3],
                 'scale_pos_weight': 1, 'eta': tempPara[0], 'silent ':False}
        num_round = 100
        data_test = xgb.DMatrix(x_test, label=y_test[:, i].reshape(-1, 1))  # 按列训练，分30次训练
        watchlist = [(data_test, 'eval'), (data_train, 'train')]
        bst = xgb.train(param, data_train, num_round,watchlist,obj=mapeobj,feval=mape)
        pre_data = xgb.DMatrix(preX)
        tempPre = bst.predict(pre_data).reshape(-1, 1)
        if i == 0:
            Y_pre = tempPre
        else:
            Y_pre = np.c_[Y_pre, tempPre]
    Y_pre = Y_pre.reshape(-1, 1)
    return Y_pre

def xgb_Fit(knownX,knownY,preX):
    xlf = xgb.XGBRegressor(max_depth=7,#11
                           learning_rate=0.06,#0.01
                           n_estimators=1000,
                           silent=True,
                           objective=mapeobj,
                           gamma=0,
                           min_child_weight=5,
                           max_delta_step=0,
                           subsample=1,#0.8
                           colsample_bytree=0.8,
                           colsample_bylevel=1,
                           reg_alpha=1e0,
                           reg_lambda=0,
                           scale_pos_weight=1,
                           seed=1850,
                           missing=None)
    x_train, x_test, y_train, y_test = train_test_split(knownX, knownY, test_size=0.5, random_state=1)
    for i in range(y_train.shape[1]):
        xlf.fit(x_train, y_train[:, i].reshape(-1,1))
        # print('Training Error: {:.3f}'.format(1 - xlf.score(x_train,y_train[:,i].reshape(-1,1))))
        # print('Validation Error: {:.3f}'.format(1 - xlf.score(x_test,y_test[:,i].reshape(-1,1))))
        #predict value for output
        tempPre = xlf.predict(preX).reshape(-1, 1)
        if i == 0:
            Y_pre = tempPre
        else:
            Y_pre = np.c_[Y_pre, tempPre]
    Y_pre = Y_pre.reshape(-1, 1)
    return Y_pre

#sklearn常规模型的集成
def CVAndPre(name,model,X_train,Y_train,X_pre):
    kf = KFold(n_splits=12)#CV
    avgPre = []
    isFirst = 1
    for train_index,valid_index in kf.split(X_train):
        trainX, validX = X_train[train_index], X_train[valid_index]
        trainY, validY = Y_train[train_index], Y_train[valid_index]
        try:  # x：60维的，y是30维的可直接训练
            model.fit(trainX,trainY)
            validY_pre = model.predict(validX)
            Y_pre = model.predict(X_pre).reshape(-1, 1)
        except:
            try:#x：60维的，y是1维的,分开多次进行训练
                for i in range(trainY.shape[1]):
                    model.fit(trainX, trainY[:, i].reshape(-1, 1))  # 按列训练，分30次训练
                    temp = model.predict(validX).reshape(-1,1)
                    tempPre = model.predict(X_pre).reshape(-1, 1)
                    if i ==0 :
                        validY_pre = temp
                        Y_pre = tempPre
                    else:
                        validY_pre  = np.c_[validY_pre,temp]
                        Y_pre = np.c_[Y_pre,tempPre]
                Y_pre = Y_pre.reshape(-1,1)
            except:
                return "error"
        err = my_score(validY_pre, validY)
        print(name +":error:%f"%err)
        if isFirst==1:
            avgPre = Y_pre
            isFirst = 0
        else:
            avgPre += Y_pre
    avgPre = avgPre/12
    return avgPre

path = datapath
linkDict={}
with open(path+"linkDict.json") as f:
    linkDict=json.loads(f.read())

#初始化数据，并且加特征
def initData(tensor):
    knownX, knownY, preX = splitData(tensor, 30, days)
    # standardize
    ss_X = StandardScaler()
    X_train = ss_X.fit_transform(knownX)
    X_final = ss_X.transform(preX)
    # Feature Engineering
    X_train = feature_extraction(X_train)
    X_final = feature_extraction(X_final)

    return X_train,knownY,X_final

#核心的处理过程
def processing(c_para):
    modelList,namelist = get_model_list()
    pre_value={}#存储每个link的预测值
    for id in range(1,133):
        pre_value[str(id)]=[]
    files = os.listdir(rootpath)
    for i in files:
        tensor = np.loadtxt(rootpath+ i + "\\" + "tensor_fill.csv", delimiter=',')
        # X_train, knownY, X_final = initData(tensor)#添加一些特征
        X_train, knownY, X_final = splitData(tensor, 30, days)#未添加任何特征
        tempValue = []
        cnt = 0
        for model_idx in range(len(modelList)):
            model = SVR(kernel='rbf', C=c_para, gamma='auto')
            pre_y = CVAndPre(namelist[model_idx], model, X_train, knownY, X_final)
            # pre_y = CVAndPre(namelist[model_idx],modelList[model_idx],X_train,knownY,X_final)
            # pre_y = xgb_Train(X_train,knownY,X_final,c_para)
            # pre_y = xgb_Fit(X_train,knownY,X_final)
            if pre_y == "error":
                continue
            if cnt==0:
                tempValue = pre_y
            else:
                tempValue += pre_y
            cnt += 1
        pre_value[i] = (tempValue/cnt).reshape(1,-1)[0]
    return pre_value

#下面的代码主要是将模型训练后预测的结果输出到最终提交文件中
def outputResult(pre_value,outputFile):
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
    for i in range(len(timeDay)):
        for j in range(len(timeMin)-1):
            timeId.append("["+timeDay[i]+" "+timeMin[j]+","+timeDay[i]+" "+timeMin[j+1]+")")
    #用于自验证时的输出
    with open(path+"selfValidNew\\"+outputFile,"w") as f:
        for i in range(len(linkDict)):
            for j in range(len(timeId)):
                f.write(linkDict[str(i+1)] + "#" + timeId[j].split(" ")[0][1:] + "#" + timeId[j] + "#" + str(pre_value[str(i+1)][j])+"\n")

if __name__ == '__main__':

    outputFile = "self_xgbModel814.txt"
    para = 0.3
    # para = [0.06, 7, 1, 0.8]
    pre_value = processing(para)
    outputResult(pre_value, outputFile)
    # #本地自验证的MAPE输出
    Rmape = SelfValidMAPE.processingOut(outputFile)
    os.remove(path + "selfValidNew\\" + outputFile)

    #xgb模型自验证代码
    # with open(path+"selfValidNew\\xgb_3.txt","w") as f:
    #     for e in range(7,11):
    #         for m in range(3,11):
    #             # for s in range(1,11):
    #             #     for c in range(1,11):
    #             outputFile = "self_xgbModel810.txt"
    #             para = [e/100,m,1,0.8]
    #             # para = [0.06,9,1,0.8]
    #             pre_value = processing(para)
    #             outputResult(pre_value,outputFile)
    #             # #本地自验证的MAPE输出
    #             Rmape = SelfValidMAPE.processingOut(outputFile)
    #             os.remove(path+"selfValidNew\\"+outputFile)
    #             f.write("#".join(map(str,para))+" mape:%f\n"%Rmape)