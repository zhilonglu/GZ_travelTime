import datetime

path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"
linkid=[]
# #genearate time
# def dateRange(start, end, step, format):
#     strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
#     minutes = (strptime(end, format) - strptime(start, format)).days*60
#     return [strftime(strptime(start, format) + datetime.timedelta(minutes=i), format) for i in range(0, minutes, step)]
timeId=[]
timeDay=[]
timeMin=[]
startDay = datetime.datetime.strptime("2016-06-01","%Y-%m-%d")
startTime = datetime.datetime.strptime("08:00:00","%H:%M:%S")
lastTime="07:58:00"#用这个时间段的流量代替之后一个小时的所有流量值，30个值取值一样。
for i in range(0,30,1):
    endDay = startDay + datetime.timedelta(days=i)
    timeDay.append(datetime.datetime.strftime(endDay,"%Y-%m-%d"))
# print(timeDay)
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
    for j in timeDay:
        avgData[(i,j)] = 0
with open(path+"gy_contest_link_traveltime_training_data.txt") as f:
    f.readline()#skip the header
    all = f.readlines()
    for i in range(len(all)):
        values = all[i].split(";")
        idx_linkid = values[0]
        idx_day = values[1]
        idx_timeRange = values[2].split(" ")[1].split(",")[0]
        if idx_day in timeDay and idx_linkid in linkid and idx_timeRange==lastTime:
            avgData[(idx_linkid,idx_day)]=float(values[3].replace("\n",""))
outputs=[]
with open(path+"submit_2.txt","w") as f:
    for i in timeId:
        day = i.split(" ")[0][1:]
        for j in linkid:
            value = avgData[(j,day)]
            f.write(j+"#"+day+"#"+i+"#"+str(value)+"\n")