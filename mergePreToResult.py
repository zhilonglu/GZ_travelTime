<<<<<<< HEAD
import os
import datetime
import json
#主要是将FCN运行完的结果合并成一个result提交
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\tensorData\\"
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
# print(timeId)
linkDict={}
with open(path+"linkDict.json") as f:
    linkDict=json.loads(f.read())
with open("C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\submit_fcn.txt", "w") as f1:
    rootfiles = os.listdir(path)
    for k in rootfiles:
        if "tensorData" in k:
            files = os.listdir(path+k+"\\")
            for i in files:
                files_id = os.listdir(path+k+"\\"+i)
                for file in files_id:
                    if file=="tensor.csv":
                            with open(path+k+"\\"+i+"\\"+file) as f2:
                                all = f2.readlines()
                                for j in range(len(all)):
                                    # f1.write(i+"#"+timeId[j].split(" ")[0][1:]+"#"+timeId[j]+"#"+all[j])
                                    f1.write(linkDict[i] + "#" + timeId[j].split(" ")[0][1:] + "#" + timeId[j] + "#" + all[j])
=======
import os
import datetime
#主要是将FCN运行完的结果合并成一个result提交
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\tensorData\\"
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
# print(timeId)
files = os.listdir(path)
with open("C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\submit_fcn.txt", "w") as f1:
    for i in files:
        files_id = os.listdir(path+i)
        for file in files_id:
            if file=="tensor.csv":
                    with open(path+i+"\\"+file) as f2:
                        all = f2.readlines()
                        for j in range(len(all)):
                            f1.write(i+"#"+timeId[j].split(" ")[0][1:]+"#"+timeId[j]+"#"+all[j])
>>>>>>> origin/master
