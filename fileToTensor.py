import datetime
import os
import json
import fillup_xy
#transform filedata to tensor and fillup tensor
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"
startTime = datetime.datetime.strptime("06:00:00","%H:%M:%S")
startDay = datetime.datetime.strptime("2017-03-01","%Y-%m-%d")

def initVariable():
    timeMin = []
    timeDay = []
    linkid = []
    linkDict = {}
    with open(path+"new_top.txt") as f:
        f.readline()
        all = f.readlines()
        for i in range(len(all)):
            linkid.append(all[i].split(";")[0])
    linkid=list(map(int,linkid))
    for i in range(0,180,2):
        endTime = startTime + datetime.timedelta(minutes=i)
        timeMin.append(datetime.datetime.strftime(endTime,"%H:%M:%S"))
    for i in range(0,122,1):
        endDay = startDay + datetime.timedelta(days=i)
        timeDay.append(datetime.datetime.strftime(endDay,"%Y-%m-%d"))
    with open(path+"reDict.json") as f:
        linkDict=json.loads(f.read())
    return timeMin,linkid,timeDay,linkDict
def readData(infile):
    timeMin, linkid, timeDay, linkDict = initVariable()
    data_dir = {}
    for i in linkid:
        for j in timeDay:
                data_dir[(i,j)] = []
    with open(path+infile) as f:
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
    return linkid,timeMin,data_dir
def fileTotesnor_first(linkid,timeMin,data_dir):
    outTensor ={}
    for i in linkid:
        outTensor[i]=[]
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
    if not os.path.exists(path+"tensorData5\\"):
        os.mkdir(path+"tensorData5\\")
    for i in outTensor:
        outputPath = path+"tensorData5\\"+str(i)
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        sortedList = sorted(outTensor[i])
        with open(outputPath+"\\tensor.csv","w") as f:
            for k in sortedList:
                f.write(",".join(map(str, k[1])) + "\n")
        # with open(outputPath+"\\tensor_all.csv","w") as f2:
        #     for k in sortedList:
        #         value = []
        #         value.append(sum(k[1][:60]))
        #         value.append(sum(k[1][60:]))
        #         f2.write(",".join(map(str, value)) + "\n")
if __name__ == '__main__':
    linkid, timeMin, data_dir = readData("gy_contest_traveltime_training_data_second.txt")
    fileTotesnor_first(linkid,timeMin,data_dir)
    fillup_xy.fillupFirst()
    fillup_xy.fillupSecond()