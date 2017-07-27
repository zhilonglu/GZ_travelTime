<<<<<<< HEAD
import datetime
import os
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"

startTime = datetime.datetime.strptime("06:00:00","%H:%M:%S")
startDay = datetime.datetime.strptime("2016-03-01","%Y-%m-%d")
linkid=[]
with open(path+"gy_contest_link_info.txt") as f:
    f.readline()
    all = f.readlines()
    for i in range(len(all)):
        linkid.append(all[i].split(";")[0])
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
with open(path+"gy_contest_link_traveltime_training_data.txt") as f:
    with open(path+"sourceData06-09.csv","w") as f1:
        f.readline()#skip the header
        all = f.readlines()
        for i in range(len(all)):
            values = all[i].split(";")
            idx_linkid = values[0]
            idx_timeDay = values[1]
            idx_timeMin = values[2].split(" ")[1].split(",")[0]
            volume = float(values[3].replace("\n",""))
            if idx_timeMin in timeMin and idx_linkid in linkid:
                f1.write(all[i].replace(";",","))
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
for i in outTensor:
    outputPath = path+"tensorData\\"+i
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    with open(outputPath+"\\tensor.csv","w") as f:
        sortedList = sorted(outTensor[i])
        for k in sortedList:
=======
import datetime
import os
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"

startTime = datetime.datetime.strptime("06:00:00","%H:%M:%S")
startDay = datetime.datetime.strptime("2016-03-01","%Y-%m-%d")
linkid=[]
with open(path+"gy_contest_link_info.txt") as f:
    f.readline()
    all = f.readlines()
    for i in range(len(all)):
        linkid.append(all[i].split(";")[0])
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
with open(path+"gy_contest_link_traveltime_training_data.txt") as f:
    with open(path+"sourceData06-09.csv","w") as f1:
        f.readline()#skip the header
        all = f.readlines()
        for i in range(len(all)):
            values = all[i].split(";")
            idx_linkid = values[0]
            idx_timeDay = values[1]
            idx_timeMin = values[2].split(" ")[1].split(",")[0]
            volume = float(values[3].replace("\n",""))
            if idx_timeMin in timeMin and idx_linkid in linkid:
                f1.write(all[i].replace(";",","))
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
for i in outTensor:
    outputPath = path+"tensorData\\"+i
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    with open(outputPath+"\\tensor.csv","w") as f:
        sortedList = sorted(outTensor[i])
        for k in sortedList:
>>>>>>> origin/master
            f.write(",".join(map(str,k[1]))+"\n")