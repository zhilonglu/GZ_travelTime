import numpy as np
import matplotlib.pyplot as plt
import datetime


#将几个提交的结果进行比较，并画出折线图
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"
def initTime():
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
    return timeId

def initLinkId():
    linkid=[]
    with open(path+"gy_contest_link_info.txt") as f:
        f.readline()
        all = f.readlines()
        for i in range(len(all)):
            linkid.append(all[i].split(";")[0])
    return linkid

def compareFile(file1,file2,file3):
    linkid = initLinkId()
    result_dict_1={}
    result_dict_2={}
    result_dict_3={}
    fin_result_1={}
    fin_result_2 = {}
    fin_result_3={}
    for i in linkid:
        result_dict_1[i]=[]
        result_dict_2[i]=[]
        result_dict_3[i]=[]
        fin_result_1[i]=[]
        fin_result_2[i]=[]
        fin_result_3[i]=[]
    with open(path+file1) as f1:
        all = f1.readlines()
        for i in range(len(all)):
            values = all[i].replace("\n","").split("#")
            result_dict_1[values[0]].append((values[1],values[3]))
        for i in result_dict_1:
            result_dict_1[i] = sorted(result_dict_1[i])
            for j in result_dict_1[i]:
                fin_result_1[i].append(j[1])
    with open(path+file2) as f2:
        all = f2.readlines()
        for i in range(len(all)):
            values = all[i].replace("\n","").split("#")
            result_dict_2[values[0]].append((values[1],values[3]))
        for i in result_dict_2:
            result_dict_2[i] = sorted(result_dict_2[i])
            for j in result_dict_2[i]:
                fin_result_2[i].append(j[1])
    with open(path+file3) as f3:
        all = f3.readlines()
        for i in range(len(all)):
            values = all[i].replace("\n","").split("#")
            result_dict_3[values[0]].append((values[1],values[3]))
        for i in result_dict_3:
            result_dict_3[i] = sorted(result_dict_3[i])
            for j in result_dict_3[i]:
                fin_result_3[i].append(j[1])
    # x_id = np.array(range(132))
    # y = []
    # for i in fin_result_1:
    #     y.append(sum(map(lambda x: abs(float(x[0]) - float(x[1])), zip(fin_result_1[i], fin_result_2[i]))) / 900)
    # y_mape = np.array(y)
    # print(sum(y_mape))
    # plt.plot(x_id, y_mape)
    # plt.show()
    for i in linkid:
        plt.figure()
        x = np.array(range(900))
        y1 = np.array(fin_result_1[i])
        y2 = np.array(fin_result_2[i])
        y3 = np.array(fin_result_3[i])
        plt.plot(x,y1,color="blue", linewidth=2.5, linestyle="-",label=file1.split(".")[0])
        plt.plot(x,y2,color="red",  linewidth=2.5, linestyle="-",label=file2.split(".")[0])
        plt.plot(x,y3,color="green", linewidth=2.5, linestyle="-", label=file3.split(".")[0])
        plt.xlabel("time")  # X轴标签
        plt.ylabel("TrafficVolume")  # Y轴标签
        plt.title(i)  # 标题
        plt.legend(loc='upper right')
        plt.show()

if __name__ == '__main__':
    compareFile("submit_knnWeighted.txt","submit_RFWeighted.txt","submit_2.txt")
