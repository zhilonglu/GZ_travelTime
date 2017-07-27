#根据补全后的tensor,求解历史08-09时间内每段时间的占比情况
import os
import datetime
import json
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"
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
rootfiles = os.listdir(path)
with open("C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\submit_historyValueWeighted.txt", "w") as f1:
    for k in rootfiles:
        if "tensorData2" in k:
            files = os.listdir(path+k+"\\")
            for i in files:
                files_id = os.listdir(path+k+"\\"+i)
                for file in files_id:
                    if file=="tensor_fill.csv":
                            with open(path+k+"\\"+i+"\\"+file) as f2:
                                all = f2.readlines()
                                data_dict={}
                                for idx in range(92):
                                    data_dict[idx]=[]
                                    values = all[j].replace("\n","").split(",")
                                    for j in range(60,90):
                                        data_dict[idx].append(float(values[j]))
                                dat = []
                                for x in range(30):
                                    dat.append(0)
                                for j in data_dict:
                                    for y in range(30):
                                        dat[y] += data_dict[j][y]
                                r = [x / sum(dat) for x in dat]
                                f1.write(linkDict[i] + "#" + ",".join(map(str,r))+"\n")
