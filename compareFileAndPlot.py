import numpy as np
import matplotlib.pyplot as plt
import datetime
import json


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
#获取linkid对应的数字
def initLinkId_num():
    linkDict = {}
    with open(path + "reDict.json") as f:
        linkDict = json.loads(f.read())
    return linkDict
#计算一个link的MAPE
def my_score(Y):
    Y_real,Y_pred = Y[0],Y[1]
    loss = 0
    cnt = 0
    for i in range(len(Y_real)):
        if float(Y_real[i]) == 0:
            continue
        else:
            loss += abs(float(Y_pred[i])/float(Y_real[i])-1)
            cnt += 1
    return loss/cnt
def compareFile(files):
    len_file = len(files)#total file input
    linkid = initLinkId()
    linkdict = initLinkId_num()
    result_dict = []
    fin_result = []
    for i in range(len_file):
        result_dict.append({})
        fin_result.append({})
    for i in linkid:
        for j in range(len_file):
            result_dict[j][i] = []
            fin_result[j][i] = []
    for i in range(len_file):
        with open(path+"selfValid\\"+files[i]) as f1:
            all = f1.readlines()
            for j in range(len(all)):
                values = all[j].replace("\n","").split("#")
                result_dict[i][values[0]].append((values[1],values[3]))
            for idx in result_dict[i]:
                result_dict[i][idx] = sorted(result_dict[i][idx])
                for j in result_dict[i][idx]:
                    fin_result[i][idx].append(j[1])
    MAPE = {}
    for idx in range(1,133):
        MAPE[idx] = 0
    for i in linkid:
        x = np.array(range(210))
        y = []
        for j in range(len_file):
            y.append(np.array(fin_result[j][i]))
            # plotInOneFigure(linkdict[i],x,y1,y2,y3,file1,file2,file3)
        # if linkdict[i] in [54,119,72,59,65,93]:
        #     plotInOneSubFigure(linkdict[i], x, y,files)
        MAPE[linkdict[i]] = my_score(y)
        # y1_y3 = my_score(y1,y3)
    xlabel = sorted(MAPE.keys())
    ylabel = []
    for key,value in MAPE.items():
        ylabel.append(value)
    srtDict = sorted(MAPE.items(),key=lambda x: x[1],reverse=True)
    for key in srtDict:
        if key[1]>0.4:
            print(key)
    # print(sorted(MAPE.items(),key=lambda x: x[1],reverse=True))
    # print(min(MAPE.items(), key=lambda x: x[1]))
    # print(max(MAPE.items(), key=lambda x: x[1]))
    # plotMapeFigure("MAPE",xlabel,ylabel,file2)
    # plotMapeFigure(i,x,y1_y3, file3)
def plotMapeFigure(i,x,y,name):
    plt.figure()
    plt.plot(x, y, color="blue", linewidth=2.5, linestyle="-", label=name+"_MAPE")
    plt.xlabel("Linkid")  # X轴标签
    plt.ylabel("MAPE")  # Y轴标签
    plt.title(i)  # 标题
    plt.legend(loc='upper right')
    plt.show()
#将所有图像画在一张图表中
def plotInOneFigure(i,x,y,files):
    color_range = ["blue","red","green"]
    plt.figure()
    for idx in range(len(files)):
        plt.plot(x,y[idx],color=color_range[idx], linewidth=2.5, linestyle="-",label=files[idx].split(".")[0])
    plt.xlabel("time")  # X轴标签
    plt.ylabel("TrafficVolume")  # Y轴标签
    plt.title(i)  # 标题
    plt.legend(loc='upper right')
    plt.show()
#将所有图像画在不同的sub图中
def plotInOneSubFigure(i,x,y,files):
    plt.figure()
    if len(files)==2:
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        plt.sca(ax1)
        plt.plot(x, y[0], color="blue", linewidth=2.5, linestyle="-", label=files[0].split(".")[0])
        plt.ylabel("TrafficVolume")  # Y轴标签
        plt.title(i)  # 标题
        plt.legend(loc='upper right')
        plt.sca(ax2)
        plt.plot(x, y[1], color="red", linewidth=2.5, linestyle="-", label=files[1].split(".")[0])
        plt.ylabel("TrafficVolume")  # Y轴标签
        plt.legend(loc='upper right')
    else:
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        plt.sca(ax1)
        plt.plot(x,y[0],color="blue", linewidth=2.5, linestyle="-",label=files[0].split(".")[0])
        plt.ylabel("TrafficVolume")  # Y轴标签
        plt.title(i)  # 标题
        plt.legend(loc='upper right')
        plt.sca(ax2)
        plt.plot(x,y[1],color="red",  linewidth=2.5, linestyle="-",label=files[1].split(".")[0])
        plt.ylabel("TrafficVolume")  # Y轴标签
        plt.legend(loc='upper right')
        plt.sca(ax3)
        plt.plot(x,y[2],color="green", linewidth=2.5, linestyle="-", label=files[2].split(".")[0])
        plt.xlabel("time")  # X轴标签
        plt.ylabel("TrafficVolume")  # Y轴标签
        plt.legend(loc='upper right')
    plt.show()
if __name__ == '__main__':
    compareFile(["selfValid_TrueY.txt","xgb_808_1.txt"])
    #selfvalid_historyValueByday_1.0,selfvalid_SixEnsemble
    # "selfvalid_historyValueByday_1.0.txt","selfValid_lastValue_1.2.txt","xgb_808_1.txt","selfValid_KNN804.txt"
    # ,"selfValid_SVRModel808.txt","selfvalid_RNNmean807.txt","selfvaild_fcn_mean803.txt","selfvaild_fcn_median803.txt"