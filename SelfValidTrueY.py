import numpy as np
import os
import datetime
import json
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

def loadPath():
    with open("config.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"],config["selfvalidpath"]

datapath,sharepath,rootpath,selfvalidpath=loadPath()

toppath=datapath
linkDict={}

with open(toppath+"linkDict.json") as f:
    linkDict=json.loads(f.read())

path=selfvalidpath


timeId=[]
timeDay=[]
timeMin=[]
startDay = datetime.datetime.strptime("2016-05-25","%Y-%m-%d")
startTime = datetime.datetime.strptime("08:00:00","%H:%M:%S")
for i in range(0,7,1):
    endDay = startDay + datetime.timedelta(days=i)
    timeDay.append(datetime.datetime.strftime(endDay,"%Y-%m-%d"))
for i in range(0,62,2):
    endTime = startTime + datetime.timedelta(minutes=i)
    timeMin.append(datetime.datetime.strftime(endTime,"%H:%M:%S"))
for i in range(len(timeDay)):
    for j in range(len(timeMin)-1):
        timeId.append("["+timeDay[i]+" "+timeMin[j]+","+timeDay[i]+" "+timeMin[j+1]+")")

ids = os.listdir(path)
with open(toppath+"selfValid\\selfValid_TrueY.txt", "w") as f1:
    for i in ids:
        taskpath=path+i+"/"
        tensor=np.loadtxt(taskpath+"tensor.csv",delimiter=',')
        trueY=tensor[-7::,60::].reshape(-1,1)
        taskname=linkDict[i]
        for j in range(len(trueY)):
            f1.write(taskname+"#"+timeId[j].split(" ")[0][1:]+"#"+timeId[j]+"#"+str(trueY[j][0])+"\n")

with open(toppath+"selfValid\\selfValid_TrueYFill.txt", "w") as f1:
    for i in ids:
        taskpath=path+i+"/"
        tensor=np.loadtxt(taskpath+"tensor_fill.csv",delimiter=',')
        trueY=tensor[-7::,60::].reshape(-1,1)
        taskname=linkDict[i]
        for j in range(len(trueY)):
            f1.write(taskname+"#"+timeId[j].split(" ")[0][1:]+"#"+timeId[j]+"#"+str(trueY[j][0])+"\n")
