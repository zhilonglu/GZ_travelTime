import datetime
import os
import json
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"

startTime = datetime.datetime.strptime("06:00:00","%H:%M:%S")
startDay = datetime.datetime.strptime("2016-03-01","%Y-%m-%d")
linkid=[]
with open(path+"new_top.txt") as f:
    f.readline()
    all = f.readlines()
    for i in range(len(all)):
        linkid.append(all[i].split(";")[0])
linkid=list(map(int,linkid))
# print(linkid)
timeMin= []
for i in range(0,180,2):
    endTime = startTime + datetime.timedelta(minutes=i)
    timeMin.append(datetime.datetime.strftime(endTime,"%H:%M:%S"))
# print(timeMin)
timeDay=[]
for i in range(0,122,1):
    endDay = startDay + datetime.timedelta(days=i)
    timeDay.append(datetime.datetime.strftime(endDay,"%Y-%m-%d"))
# print(timeDay)
data_dir={}
for i in linkid:
    for j in timeDay:
        for k in timeMin:
            data_dir[(i,j)] = []
# print(len(data_dir))
linkDict={}
with open(path+"reDict.json") as f:
    linkDict=json.loads(f.read())
# print(linkDict)
with open(path+"gy_contest_link_traveltime_training_data.txt") as f:
        f.readline()#skip the header
        all = f.readlines()
        for i in range(len(all)):
            values = all[i].split(";")
            idx_linkid = linkDict[values[0]]
            idx_timeDay = values[1]
            idx_timeMin = values[2].split(" ")[1].split(",")[0]
            volume = float(values[3].replace("\n",""))
            if idx_timeMin in timeMin and idx_linkid in linkid:
                data_dir[(idx_linkid,idx_timeDay)].append((idx_timeMin,volume))
outTensor ={}
for i in linkid:
    outTensor[i]=[]
timeMin_data={}
for i in data_dir:
    tempList = []
    tempMin=[]
    for j in data_dir[i]:
        tempMin.append(j[0])
    for k in list(set(timeMin)-set(tempMin)):#补全缺失时间段的数据为0
        data_dir[i].append((k,0))
    sortedList = sorted(data_dir[i])
    for k in sortedList:
        tempList.append(k[1])
    outTensor[i[0]].append((i[1],tempList))
if not os.path.exists(path+"tensorData3\\"):
    os.mkdir(path+"tensorData3\\")
for i in outTensor:
    outputPath = path+"tensorData3\\"+str(i)
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    sortedList = sorted(outTensor[i])
    with open(outputPath+"\\tensor.csv","w") as f:
        for k in sortedList:
            f.write(",".join(map(str, k[1])) + "\n")
    with open(outputPath+"\\tensor_all.csv","w") as f2:
        for k in sortedList:
            value = []
            value.append(sum(k[1][:60]))
            value.append(sum(k[1][60:]))
            f2.write(",".join(map(str, value)) + "\n")
