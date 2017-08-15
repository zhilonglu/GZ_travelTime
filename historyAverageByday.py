import datetime
import json
import SelfValidMAPE

def loadPath():
    with open("config.json") as f:
    #这是用于自验证的代码
    # with open("configSelfValid.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"],config["selfvalidpath"],config["startdate"],config["days"]

datapath,sharepath,rootpath,selfvalidpath,startdate,days=loadPath()

path = datapath
linkid=[]
timeId=[]
timeDay=[]
timeMin=[]
weekday = []#星期几
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
time_range= []
for i in range(0,60,2):
    endTime = startTime + datetime.timedelta(minutes=i)
    time_range.append(datetime.datetime.strftime(endTime,"%H:%M:%S"))
with open(path+"gy_contest_link_info.txt") as f:
    f.readline()
    all = f.readlines()
    for i in range(len(all)):
        linkid.append(all[i].split(";")[0])
avgData={}
for i in linkid:
    for d in range(0,7):#代表的是星期几
        for j in time_range:
            avgData[(i,d,j)] = []
with open(path+"gy_contest_traveltime_training_data_second.txt") as f:
    f.readline()#skip the header
    all = f.readlines()
    for i in range(len(all)):
        values = all[i].split(";")
        idx_linkid = values[0]
        idx_day = datetime.datetime.strptime(values[1],"%Y-%m-%d").weekday()
        idx_timeRange = values[2].split(" ")[1].split(",")[0]
        if idx_timeRange in time_range and idx_linkid in linkid and values[1] < startdate:
            avgData[(idx_linkid,idx_day,idx_timeRange)].append(float(values[3].replace("\n","")))
for i in avgData:
    if len(avgData[i])==0:
        temp = 0
    else:
        temp = sum(avgData[i])/len(avgData[i])
    avgData[i] = temp
#注意生成selfValid文件时，修改文件的路径
#进行gridsearch找出最好的乘比例
per_range = [i/10 for i in range(10,15)]
for per in per_range:
    outputfile = "submit_historyValueByday_"+str(per)+".txt"
    with open(selfvalidpath+outputfile,"w") as f:
        for i in timeId:
            day = i.split(" ")[0][1:]
            w_day = datetime.datetime.strptime(day, "%Y-%m-%d").weekday()
            for j in linkid:
                value = avgData[(j,w_day,i.split(" ")[1].split(",")[0])]*per
                f.write(j+"#"+day+"#"+i+"#"+str(value)+"\n")
    # SelfValidMAPE.processingOut(outputfile)