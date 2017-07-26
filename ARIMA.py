from statsmodels.graphics.api import qqplot
from scipy import  stats
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime
import os

path = 'C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\'
linkid=[]
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
timeId=[]
for j in range(len(timeMin)-1):
    for i in range(len(timeDay)):
        timeId.append("["+timeDay[i]+" "+timeMin[j]+","+timeDay[i]+" "+timeMin[j+1]+")")
# print(timeId)
file_dict ={}
#file_dict[id]=[(time,value)]
for id in linkid:
    file_dict[id]=[]
sort_file_dict={}
def process():
    dataPath = path+"tensorData\\"
    files = os.listdir(dataPath)
    pre_result={}
    for i in files:
        arima_data = {}
        pre_result[i]=[]
        for n in range(30):
            arima_data[n] = []
        files_id = os.listdir(dataPath + i)
        for file in files_id:
            if file == "tensor_fill.csv":
                with open(dataPath + i + "\\" + file) as f2:
                    all = f2.readlines()
                    for j in range(0, len(all) - 30):
                        tempdata = list(map(float, all[j].replace("\n", "").split(",")))
                        for k in range(60, 90):
                            arima_data[k - 60].append(tempdata[k])
        for idx in arima_data:
            pre_result[i].append((idx,ar_model(arima_data[idx])))
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
def main():
    result = process()
    with open(path + "submit_ARIMA.txt", "w") as f:
        for k in result:
            value_list = result[k]
            preV=[]
            for i in value_list:
                preV +=i[1]
            for j in range(len(timeId)):
                day = timeId[j].split(" ")[0][1:]
                f.write(k + "#" + day + "#" + timeId[j] + "#" + str(preV[j]) + "\n")
if __name__ == '__main__':
    main()
