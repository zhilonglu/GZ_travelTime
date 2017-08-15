import datetime
import json
import os
import SelfValidMAPE

def loadPath():
    with open("config.json") as f:
    #这是用于自验证的代码
    # with open("configSelfValid.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"],config["selfvalidpath"],config["startdate"],config["days"]

datapath,sharepath,rootpath,selfvalidpath,startdate,days=loadPath()

linkDict={}
with open(datapath+"linkDict.json") as f:
    linkDict=json.loads(f.read())
reDict={}
with open(datapath+"reDict.json") as f:
    reDict=json.loads(f.read())
linkid = os.listdir(rootpath)
timeId=[]
timeDay=[]
timeMin=[]
startDay = datetime.datetime.strptime(startdate,"%Y-%m-%d")
startTime = datetime.datetime.strptime("08:00:00","%H:%M:%S")
lastTime="07:58:00"#用这个时间段的流量代替之后一个小时的所有流量值，30个值取值一样。
for i in range(0,int(days),1):
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
avgData={}
for i in linkid:
    for j in timeDay:
        avgData[(linkDict[i],j)] = 0
with open(datapath+"gy_contest_traveltime_training_data_second.txt") as f:
    f.readline()#skip the header
    all = f.readlines()
    for i in range(len(all)):
        values = all[i].split(";")
        idx_linkid = values[0]
        idx_day = values[1]
        idx_timeRange = values[2].split(" ")[1].split(",")[0]
        if idx_day in timeDay and str(reDict[idx_linkid]) in linkid and idx_timeRange==lastTime:
            avgData[(idx_linkid,idx_day)]=float(values[3].replace("\n",""))
outputs=[]
per_range = [i/10 for i in range(10,16)]
for per in per_range:
    with open(datapath+"resultNew\\submit_lastValue_"+str(per)+".txt","w") as f:
    #如果进行本地验证时，则启用下述代码
    # with open(datapath + "selfValid\\selfValid_lastValue1_new.txt", "w") as f:
        for i in linkid:
            for j in timeId:
                day = j.split(" ")[0][1:]
                value = avgData[(linkDict[i],day)]*per
                f.write(linkDict[i] + "#" + day + "#" + j + "#" + str(value) + "\n")

#进行gridsearch找出最好的乘比例
# per_range = [i/10 for i in range(10,20)]
# for per in per_range:
#     filename = "selfValid_lastValue_"+str(per)+".txt"
#     with open(datapath + "selfValid\\"+filename, "w") as f:
#         for i in linkid:
#             for j in timeId:
#                 day = j.split(" ")[0][1:]
#                 value = avgData[(linkDict[i], day)]*per
#                 f.write(linkDict[i] + "#" + day + "#" + j + "#" + str(value) + "\n")
#     SelfValidMAPE.processingOut(filename)