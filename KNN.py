import datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import os
import json

# #KNN进行预测的结果
# def KNN_pre(k,i):
#     tensor = np.loadtxt(path + k + "\\" + i + "\\" + "tensor_fill.csv", delimiter=',')
#     knownX, knownY, preX = splitData(tensor, 30, 30)
#     for tm in range(10):
#         kf = KFold(n_splits=int(knownX.shape[0]))
#         for train_index, valid_index in kf.split(knownX):
#             trainX, validX = knownX[train_index], knownX[valid_index]
#             T_x, V_x = trainX.sum(axis=1).reshape(-1, 1), validX.sum(axis=1).reshape(-1, 1)
#             trainY, validY = knownY[train_index], knownY[valid_index]
#             T_y, V_y = trainY.sum(axis=1).reshape(-1, 1), validY.sum(axis=1).reshape(-1, 1)
#             neigh = KNeighborsRegressor(n_neighbors=9)
#             neigh.fit(T_x, T_y)
#     P_x = preX.sum(axis=1).reshape(-1, 1)
#     P_y = neigh.predict(P_x)
#     pre_y = P_y.T[0]
#     # 计算08-09各自的比例情况
#     per_tensor = np.loadtxt(path + k + "\\" + i + "\\" + "tensor.csv", delimiter=',')
#     rateData = per_tensor[:92, 60:].sum(axis=0).reshape(1, -1)
#     rate = rateData / rateData.sum()
#     r = rate[0]
#     # 分摊比例，计算每两分钟的预测值
#     for idx_y in pre_y:
#         pre_value[i] += [idx_y * v for v in r]
#     return r,pre_y
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
            r = []  # 08-09每两分钟分配的比例
            pre_y = []#KNN 预测的08-09的总值
            for file in files_id:
                if file == "tensor.csv":
                    with open(path + k + "\\" + i + "\\" + file) as f2:
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
                    with open(path + k + "\\" + i + "\\" + file) as f2:
                        all = f2.readlines()
                        x=[]
                        y=[]
                        pre_x=[]
                        for j in range(0,92):
                            values = all[j].replace("\n","").split(",")
                            x.append([float(values[0])])
                            y.append(float(values[1]))
                        for j in range(92,122):
                            values = all[j].replace("\n", "").split(",")
                            pre_x.append([float(values[0])])
                        #KNN sklearn model
                        # neigh = KNeighborsRegressor(n_neighbors=5)
                        # neigh.fit(x, y)
                        # pre_y = neigh.predict(pre_x)
                        #RF sklearn model
                        model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2,
                                                      random_state=0)
                        model.fit(x,y)
                        pre_y = model.predict(pre_x)
                    f2.close
            for idx_y in pre_y:
                pre_value[i] += [idx_y*v for v in r]
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
with open(path+"submit_RFWeighted.txt","w") as f:
    output=[]
    for idx in pre_value:
        output += pre_value[idx]
    for i in range(len(linkDict)):
        for j in range(len(timeId)):
            f.write(linkDict[str(i+1)] + "#" + timeId[j].split(" ")[0][1:] + "#" + timeId[j] + "#" + str(output[i*900+j])+"\n")