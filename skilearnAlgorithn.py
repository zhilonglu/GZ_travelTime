import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import gaussian_process
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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

def get_model_list():
    model_list, name_list = [], []

    model_list.append(gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1))
    name_list.append('GaussianProcess')

    model_list.append(KNeighborsRegressor(weights = 'uniform',n_neighbors=24))
    name_list.append('KNN_unif')

    model_list.append(KNeighborsRegressor(weights = 'distance',n_neighbors=24))
    name_list.append('KNN_dist')

    model_list.append(SVR(kernel = 'poly', C = 1, gamma = 'auto', coef0 = 0, degree = 2))
    name_list.append('SVR_poly')

    model_list.append(SVR(kernel = 'rbf', C = 1, gamma = 'auto', coef0 = 0, degree = 2))
    name_list.append('SVR_rbf')

    model_list.append(DecisionTreeRegressor())
    name_list.append('DT')

    model_list.append(RandomForestRegressor(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0))
    name_list.append('RF')

    model_list.append(ExtraTreesRegressor(n_estimators=100, max_depth=None, max_features='auto', min_samples_split=2, random_state=0))
    name_list.append('ET')

    return model_list,name_list

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    return acc_rate

#xgboost进行预测的结果
def xgb_pre(k,i):
    tensor = np.loadtxt(path + k + "\\" + i + "\\" + "tensor_fill.csv", delimiter=',')
    knownX, knownY, preX = splitData(tensor, 30, 30)
    x_train, x_test, y_train, y_test = train_test_split(knownX, knownY, test_size=0.5, random_state=1)
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'reg:logistic','eval_metric':'map'}
    bst = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)
    y_hat = bst.predict(data_test)
    xgb_rate = show_accuracy(y_hat, y_test, 'XGBoost ')
    print('XGBoost：%.3f%%' % xgb_rate)
    # return pre_y

#MAPE
def my_score(Y_real, Y_pred):
    MAPE = np.mean(np.abs(np.ones_like(Y_pred) - Y_pred / Y_real))
    return np.array([MAPE]).reshape(-1, 1)

def CVAndPre(name,model,X_train,Y_train,X_pre):
    kf = KFold(n_splits=12)#CV
    avgPre = []
    isFirst = 1
    try:
        for train_index,valid_index in kf.split(X_train):
            trainX, validX = X_train[train_index], X_train[valid_index]
            trainY, validY = Y_train[train_index], Y_train[valid_index]
            model.fit(trainX,trainY)
            validY_pre = model.predict(validX)
            err = my_score(validY_pre,validY)
            # print(name + ":error:%f"%err)
            Y_pre = model.predict(X_pre).reshape(-1,1)
            if isFirst==1:
                avgPre = Y_pre
                isFirst = 0
            else:
                avgPre += Y_pre
        avgPre = avgPre/12
        return avgPre
    except:
        return "error"

path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"
linkDict={}
with open(path+"linkDict.json") as f:
    linkDict=json.loads(f.read())

#核心的处理过程
def processing():
    modelList,namelist = get_model_list()
    rootfiles = os.listdir(path)
    pre_value={}#存储每个link的预测值
    for id in range(1,133):
        pre_value[str(id)]=[]
    for k in rootfiles:
        if "tensorData3" in k:
            files = os.listdir(path + k + "\\")
            for i in files:
                tensor = np.loadtxt(path + k + "\\" + i + "\\" + "tensor_fill.csv", delimiter=',')
                knownX, knownY, preX = splitData(tensor, 30, 30)
                xgb_pre(k,i)
                # tempValue = []
                # cnt = 0
                # for model_idx in range(len(modelList)):
                #     pre_y = CVAndPre(namelist[model_idx],modelList[model_idx],knownX,knownY,preX)
                #     if pre_y=="error":
                #         continue
                #     if cnt==0:
                #         tempValue = pre_y
                #     else:
                #         tempValue += pre_y
                #     cnt += 1
                # pre_value[i] = (tempValue/cnt).reshape(1,-1)[0]
    return pre_value


#下面的代码主要是将模型训练后预测的结果输出到最终提交文件中
def outputResult(pre_value):
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
    with open(path+"result\\submit_AllModelMean2.txt","w") as f:
        for i in range(len(linkDict)):
            for j in range(len(timeId)):
                f.write(linkDict[str(i+1)] + "#" + timeId[j].split(" ")[0][1:] + "#" + timeId[j] + "#" + str(pre_value[str(i+1)][j])+"\n")

if __name__ == '__main__':
    np.random.seed(0)
    pre_value = processing()
    # outputResult(pre_value)