#合并几个文件成一个文件
import SelfValidMAPE
import os
import json
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\result\\"
#进行selfValid时启用的代码
valid_path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\selfValid\\"

datapath = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"
linkDict={}
with open(datapath+"linkDict.json") as f:
    linkDict=json.loads(f.read())

#找出数组中倒数第二小的非0元素
def findSecondMinExceptZero(a):
    tmp = sorted(a)
    # return max(tmp)
    for i in range(len(tmp)):
        if tmp[i] != 0:
            return tmp[i+1]
    return 0
#集成所有输入，并得出最终的一个输出
def integer(files,outputfile):
    result_dict=[]
    for idx in range(len(files)):
        result_dict.append({})
        with open(path+files[idx]) as f1:
            all = f1.readlines()
            for i in range(len(all)):
                values = all[i].replace("\n","").split("#")
                result_dict[idx][(values[0],values[1],values[2])] = float(values[3])
    with open(path+outputfile,"w") as f4:
        for i in result_dict[0]:
            tempList = []
            for j in range(len(files)):
                tempList.append(result_dict[j][i])
            f4.write("#".join(i)+"#"+str(min(tempList))+"\n")

#传人一个字典，其中存储的是某个文件的KV，计算其与真实值的单个link MAPE
def my_score(result_dict):
    file_error_dict = {}  # 存储所有文件的单link误差
    for key in linkDict:
        file_error_dict[linkDict[key]] = []
    with open(valid_path+"selfValid_TrueY.txt") as f:
        f_all = f.read()
        lines = f_all.split("\n")
        for line in lines:
            ls = line.split("#")
            if (len(ls) == 4):
                trueV = float(ls[3])
                preV = result_dict[(ls[0], ls[1], ls[2])]
                if trueV == 0:
                    continue
                else:
                    temploss = abs(trueV - preV) / trueV
                    file_error_dict[ls[0]].append(temploss)
    for key in file_error_dict:
        file_error_dict[key] = sum(file_error_dict[key])/len(file_error_dict[key])
    return file_error_dict
# 自动集成所有结果中误差最小的结果
def EnsembleMinError(files_valid,files_submit,outputfile):
    submit_result_dict = []#存储所有文件的结果
    valid_result_dict = []#存储所有文件的结果，验证的
    file_error_dict = []#存储所有文件的单link误差
    for idx in range(len(files_valid)):#读取所有的文件结果
        valid_result_dict.append({})
        file_error_dict.append({})
        with open(valid_path + files_valid[idx]) as f1:
            all = f1.readlines()
            for i in range(len(all)):
                values = all[i].replace("\n", "").split("#")
                valid_result_dict[idx][(values[0], values[1], values[2])] = float(values[3])
        file_error_dict[idx] = my_score(valid_result_dict[idx])
    for idx in range(len(files_submit)):#读取所有的文件结果
        submit_result_dict.append({})
        with open(path + files_submit[idx]) as f1:
            all = f1.readlines()
            for i in range(len(all)):
                values = all[i].replace("\n", "").split("#")
                submit_result_dict[idx][(values[0], values[1], values[2])] = float(values[3])
    linkid = []#存储的是所有的link
    for key in linkDict:
        linkid.append(linkDict[key])
    with open(path + outputfile, "w") as f:
        for lk in linkid:
            tempValue = {}
            for key in range(len(file_error_dict)):
                tempValue[key] = file_error_dict[key][lk]
            srtDict = sorted(tempValue.items(), key=lambda x: x[1])#list:(key,value)
            for j in submit_result_dict[srtDict[0][0]]:
                outputTemp = submit_result_dict[srtDict[0][0]][j]
                if j[0] == lk:
                    f.write("#".join(j) + "#" + str(outputTemp) + "\n")
outputfile = "submit_SixEnsembleRNN808_808.txt"
# integer(["submit_rnn_mean808.txt","submit_lastValue1.5.txt","knn_804.txt","submit_fcn_mean5.txt"
#          ,"SVR_rbf0.3Model808.txt","submit_historyValueByday1.0.txt"],outputfile)
submit_file = ["submit_rnn_mean808.txt","submit_fcn_mean5.txt","SVR_rbf0.3Model808.txt","knn_804.txt",
                  "submit_historyValueByday1.0.txt","submit_lastValue1.5.txt","submit_xgbModel808.txt","submit_rnn_median808.txt",
                    "submit_fcn_median5.txt", "submit_SVRAndknnRNN808fcnmean5max.txt",
                  "submit_SVRRNN808knnAndHV1.0.txt",outputfile]
selfValid_file = ["sv_rnn_mean808.txt","selfvaild_fcn_mean803.txt","selfValid_SVRModel808.txt","selfValid_KNN804.txt",
         "selfvalid_historyValueByday_1.0.txt","selfValid_lastValue_1.5.txt","xgb_808_1.txt","sv_rnn_median808.txt",
                 "selfvaild_fcn_median803.txt", "selfvalid_SVRAnsknnRNN808fcnmean_803max.txt",
                  "selfvalid_SVRRNN808knn_804AndHV1.0.txt","selfvalid_SixEnsembleRNN808.txt"]
#selfvalid_RNNmean807 sv_rnn_mean808
outputFile3 = "submit_EnsembleAllRnn808_808_2.txt"
EnsembleMinError(selfValid_file,submit_file,outputFile3)