import datetime
from sklearn import gaussian_process
from sklearn.neighbors import KNeighborsRegressor
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

def loadPath():
    # with open("config.json") as f:
    ##这是用于自验证的代码
    with open("configSelfValid.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"],config["selfvalidpath"],config["startdate"],config["days"]
datapath,sharepath,rootpath,selfvalidpath,startdate,days=loadPath()

#将读取的tensor进行拆分，分为训练部分和测试部分
def splitData(tensor,n_output,n_pred):
    n_known=tensor.shape[0]-n_pred
    n_input=tensor.shape[1]-n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known+n_pred, 0: n_input]
    return (knownX,knownY,preX)

def get_model_list():
    model_list, name_list = [], []

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
    model_list.append(SVR(kernel = 'rbf', C = 0.5, gamma = 'auto'))
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

#MAPE
def my_score(Y_real, Y_pred):
    MAPE = np.mean(np.abs(np.ones_like(Y_pred) - Y_pred / Y_real))
    return np.array([MAPE]).reshape(-1, 1)


#XGB利用gridsearch来进行参数的选择
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

#xgboost进行预测的结果
def xgb_Train(knownX,knownY,preX,tempPara):
    #max_depth,subsample,colsample_bytree,eta 3,1.0,0.8,0.05 0.507587
    # max_depth,subsample,colsample_bytree,eta 6,0.8,0.8,0.15 0.510138
    x_train, x_test, y_train, y_test = train_test_split(knownX, knownY, test_size=0.5, random_state=1)
    for i in range(y_train.shape[1]):
        data_train = xgb.DMatrix(x_train, label=y_train[:, i].reshape(-1, 1))# 按列训练，分30次训练
        # data_test = xgb.DMatrix(x_test, label=y_test[:, i].reshape(-1, 1))
        param = { 'n_estimators': 1000, 'max_depth': tempPara[1],
                 'min_child_weight': 5, 'gamma': 0, 'subsample': tempPara[2], 'colsample_bytree': tempPara[3],
                 'scale_pos_weight': 1, 'eta': tempPara[0], 'silent ':1,'objective': 'count:poisson'}#“reg:linear”
        num_round = 100
        # evallist = [(data_test, 'eval'), (data_train, 'train')]
        bst = xgb.train(param, data_train, num_round)
        pre_data = xgb.DMatrix(preX)
        tempPre = bst.predict(pre_data).reshape(-1, 1)
        #tempPre = bst.predict(pre_data,ntree_limit=bst.best_ntree_limit).reshape(-1,1)
        if i == 0:
            Y_pre = tempPre
        else:
            Y_pre = np.c_[Y_pre, tempPre]
    Y_pre = Y_pre.reshape(-1, 1)
    return Y_pre

#评价函数
def mape(Y_real, Y_pred):
    loss = 0
    cnt = 0
    for i in range(len(Y_real)):
        if float(Y_real[i]) == 0:
            continue
        else:
            loss += abs(float(Y_pred[i])/float(Y_real[i])-1)
            cnt += 1
    return "mape",loss/cnt

#xgboost进行预测的结果2
def xgb_Fit(knownX,knownY,preX):
    xlf = xgb.XGBRegressor(max_depth=11,
                           learning_rate=0.01,
                           n_estimators=301,
                           silent=True,
                           objective=mape,
                           gamma=0,
                           min_child_weight=5,
                           max_delta_step=0,
                           subsample=0.8,
                           colsample_bytree=0.8,
                           colsample_bylevel=1,
                           reg_alpha=1e0,
                           reg_lambda=0,
                           scale_pos_weight=1,
                           seed=9,
                           missing=None)
    x_train, x_test, y_train, y_test = train_test_split(knownX, knownY, test_size=0.5, random_state=1)
    for i in range(y_train.shape[1]):
        xlf.fit(x_train, y_train[:, i].reshape(-1, 1), eval_metric=mape, verbose=False)
                # eval_set=[(x_test, y_test[:, i].reshape(-1, 1))], early_stopping_rounds=2)
        tempPre = xlf.predict(preX).reshape(-1, 1)
        if i == 0:
            Y_pre = tempPre
        else:
            Y_pre = np.c_[Y_pre, tempPre]
    Y_pre = Y_pre.reshape(-1, 1)
    return Y_pre

#对model进行gridsearch
def gridSearach(X_train,Y_train):
    tuned_parameters = {'kernel': ['rbf'],
                         'C': [0,1,0.1]}
    scores = ['precision', 'recall']
    for score in scores:
        clf = GridSearchCV(SVR(C=1), tuned_parameters, cv=12,
                           scoring='%s_macro' % score)
        clf.fit(X_train, Y_train)
        print(clf.best_params_)


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
        # err = my_score(validY_pre, validY)
        # print(name + ":error:%f"%err)
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

#核心的处理过程
def processing(tempParam):
    modelList,namelist = get_model_list()
    pre_value={}#存储每个link的预测值
    for id in range(1,133):
        pre_value[str(id)]=[]
    files = os.listdir(selfvalidpath)
    for i in files:
        tensor = np.loadtxt(selfvalidpath+ i + "\\" + "tensor_fill.csv", delimiter=',')
        knownX, knownY, preX = splitData(tensor, 30, days)
        tempValue = []
        cnt = 0
        for model_idx in range(len(modelList)):
            # pre_y = CVAndPre(namelist[model_idx],modelList[model_idx],knownX,knownY,preX)
            # model = SVR(kernel='rbf', C=c_para, gamma='auto')
            # pre_y = CVAndPre(namelist[model_idx],model,knownX,knownY,preX)
            pre_y = xgb_Train(knownX,knownY,preX,tempParam)
            # pre_y = xgb_Fit(knownX, knownY, preX)
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
    with open(path+"selfValid\\"+outputFile,"w") as f:
    # with open(path + "result\\"+outputFile, "w") as f:
        for i in range(len(linkDict)):
            for j in range(len(timeId)):
                f.write(linkDict[str(i+1)] + "#" + timeId[j].split(" ")[0][1:] + "#" + timeId[j] + "#" + str(pre_value[str(i+1)][j])+"\n")

if __name__ == '__main__':
    # with open(path+"selfValid\\xgb_model.txt","w") as f:
        # for e in range(0,201):
        #     for m in range(3,11):
        #         for s in range(1,11):
        #             for c in range(1,11):
        outputFile = "xgb_808_1.txt"
        # para = [e/100,m,s/10,c/10]
        para = [0.05,3,1,0.8]
        pre_value = processing(para)
        outputResult(pre_value,outputFile)
        # #本地自验证的MAPE输出
        Rmape = SelfValidMAPE.processingOut(outputFile)
        # os.remove(path+"selfValid\\"+outputFile)
        # f.write("#".join(map(str,para))+" mape:%f\n"%Rmape)