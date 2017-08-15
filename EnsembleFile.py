#合并几个文件成一个文件
import SelfValidMAPE
import os
import json
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\resultNew\\"
#进行selfValid时启用的代码
# path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\selfValid\\"

datapath = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"
linkDict={}
with open(datapath+"linkDict.json") as f:
    linkDict=json.loads(f.read())

#找出数组中最小的非0元素
def findMinExceptZero(a):
    tmp = sorted(a)
    for i in tmp:
        if i != 0:
            return i
    return 0
#找出数组中倒数第二小的非0元素
def findSecondMinExceptZero(a):
    tmp = sorted(a)
    # return max(tmp)
    for i in range(len(tmp)):
        if tmp[i] != 0:
            return tmp[i+1]
    return 0
#根据不同的比例集成最终结果
def weightedEnsemble(a):
    if len(a)==2:
        return a[0]*0.95+a[1]*0.05
    else:
        return a[0]*0.9+a[1]*0.05+a[2]*0.05
#集成所有输入，并得出最终的一个输出
def integer(files,outputfile):
    result_dict=[]
    for idx in range(len(files)):
        result_dict.append({})
        with open(path+files[idx]) as f1:
            all = f1.readlines()
            for i in range(len(all)):
                values = all[i].replace("\n","").split("#")
                result_dict[idx][(values[0],values[1],values[2])] = float(values[3])*0.9999
    with open(path+outputfile,"w") as f4:
        for i in result_dict[0]:
            tempList = []
            for j in range(len(files)):
                tempList.append(result_dict[j][i])
            f4.write("#".join(i)+"#"+str(findSecondMinExceptZero(tempList))+"\n")

# 集成所有的数据，并其中的linkid替换为一个文件对应的数据，最终输出文件
#linkid为需要进行替换的link，replacefile为需要进行替换的文件
def RepalceAndEnsemble(files, outputfile,replace_dict):
    result_dict = []
    linkList = {}#key:111.txt value={}存放结果数据
    for key in replace_dict:
        linkid = [linkDict[str(x)] for x in replace_dict[key]]
        replace_dict[key] = linkid
        linkList[key] = {}
        with open(path+key) as f:
            all = f.readlines()
            for i in range(len(all)):
                values = all[i].replace("\n", "").split("#")
                linkList[key][(values[0], values[1], values[2])] = float(values[3])
    print(replace_dict)
    for idx in range(len(files)):
        result_dict.append({})
        with open(path + files[idx]) as f1:
            all = f1.readlines()
            for i in range(len(all)):
                values = all[i].replace("\n", "").split("#")
                result_dict[idx][(values[0], values[1], values[2])] = float(values[3])
    cnt = 0
    with open(path + outputfile, "w") as f4:
        for i in result_dict[0]:
            tempList = []
            for j in range(len(files)):
                tempList.append(result_dict[j][i])
            outputTemp = min(tempList)
            for key in replace_dict:
                if i[0] in replace_dict[key]:
                    cnt += 1
                    outputTemp = linkList[key][i]
            f4.write("#".join(i) + "#" + str(outputTemp) + "\n")
    print(str(cnt))
#传人一个字典，其中存储的是某个文件的KV，计算其与真实值的单个link MAPE
def my_score(result_dict):
    file_error_dict = {}  # 存储所有文件的单link误差
    for key in linkDict:
        file_error_dict[linkDict[key]] = []
    with open(path+"selfValid_TrueY.txt") as f:
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
def EnsembleMinError(files,outputfile):
    result_dict = []#存储所有文件的结果
    file_error_dict = []#存储所有文件的单link误差
    for idx in range(len(files)):#读取所有的文件结果
        result_dict.append({})
        file_error_dict.append({})
        with open(path + files[idx]) as f1:
            all = f1.readlines()
            for i in range(len(all)):
                values = all[i].replace("\n", "").split("#")
                result_dict[idx][(values[0], values[1], values[2])] = float(values[3])
        file_error_dict[idx] = my_score(result_dict[idx])
    linkid = []#存储的是所有的link
    for key in linkDict:
        linkid.append(linkDict[key])
    with open(path + outputfile, "w") as f:
        for lk in linkid:
            tempValue = {}
            for key in range(len(file_error_dict)):
                tempValue[key] = file_error_dict[key][lk]
            srtDict = sorted(tempValue.items(), key=lambda x: x[1])
            for j in result_dict[srtDict[0][0]]:
                outputTemp = result_dict[srtDict[0][0]][j]
                if j[0] == lk:
                    f.write("#".join(j) + "#" + str(outputTemp) + "\n")
outputfile = "submit_SixEnsembleFinal.txt"
integer(["submit_FCNmean811.txt","submit_lastValue_1.5.txt","knn.txt","submit_rnnmean812.txt"
         ,"submit_svrModell810.txt","submit_historyValueByday_1.0.txt"],outputfile)
# outputfile = "submit_SVRRNN807knnAndHV1.0.txt"
# integer(["SVR_rbf0.3Model808.txt","submit_rnn_mean807.txt","knn_804.txt","submit_historyValueByday1.0.txt"],outputfile)
# submit_file = ["submit_rnn_mean807.txt","submit_fcn_mean5.txt","SVR_rbf0.3Model808.txt","knn_804.txt",
#                   "submit_historyValueByday1.0.txt","submit_lastValue1.5.txt","submit_xgbModel808.txt","submit_rnn_median808.txt",
#                     "submit_fcn_median5.txt", "submit_SVRAndknnRNN807fcnmean5max.txt",
#                   "submit_SVRRNN807knnAndHV1.0.txt",outputfile]
# selfValid_file = ["sv_rnn_mean808.txt","selfvaild_fcn_mean803.txt","selfValid_SVRModel808.txt","selfValid_KNN804.txt",
#          "selfvalid_historyValueByday_1.0.txt","selfValid_lastValue_1.5.txt","xgb_808_1.txt","sv_rnn_median808.txt",
#                  "selfvaild_fcn_median803.txt", "selfvalid_SVRAndknn_802RNN807fcnmean_803max.txt",
#                   "selfvalid_SVRRNN807knn_802AndHV1.0.txt","selfvalid_SevenEnsemble1.txt"]
# outputFile3 = "submit_EnsembleAllRnn807_808.txt"
# EnsembleMinError(submit_file,selfValid_file,outputFile3)
#下面是生成自验证的本地结果的代码
# 进行gridsearch找出最好的乘比例
# per_LVrange = [i/10 for i in range(10,20)]
# per_HVrange = [i/10 for i in range(10,15)]
# for perLV in per_LVrange:
#     for perHV in per_HVrange:
#         outputFile = "selfvalid_SVRAndfcnmean5AndKNN803LV_"+str(perLV)+"HV_"+str(perHV)+".txt"
#         integer("selfvalid_SVRAndfcnmean5AndKNN803.txt","selfvalid_historyValueByday_"+str(perHV)+".txt",
#                 "selfValid_lastValue_"+str(perLV)+".txt",outputFile)
#         SelfValidMAPE.processingOut(outputFile)
#第一步集成结果
# outputFile = "selfvalid_SixEnsembleRNN809.txt"
# integer(["sv_rnn_mean808.txt","selfValid_SVRModel808.txt","selfValid_KNN804.txt","selfvalid_historyValueByday_1.0.txt"],outputFile)
# SelfValidMAPE.processingOut(outputFile)
# 第二步集成
# outputFile3 = "selfvalid_Ensemble_t4.txt"
# EnsembleMinError(["sv_rnn_mean808.txt","selfvaild_fcn_mean803.txt","selfValid_SVRModel808.txt","selfValid_KNN804.txt",
#          "selfvalid_historyValueByday_1.0.txt","selfValid_lastValue_1.5.txt","xgb_808_1.txt","sv_rnn_median808.txt",
#                  "selfvaild_fcn_median803.txt", outputFile, "selfvalid_SVRAndknn_802RNN807fcnmean_803max.txt",
#                   "selfvalid_SVRRNN807knn_802AndHV1.0.txt"],outputFile3)
# # outputFile,"selfvalid_SVRAndknn_802RNN807fcnmean_803max.txt","selfvalid_SVRRNN807knn_802AndHV1.0.txt",sv_rnn_mean808
# SelfValidMAPE.processingOut(outputFile3)
#在第一次的结果上进行第二次集成
# outputFile2 = "selfvalid_SixEnsemble_t1.txt"
# replace_dict ={'xgb_808_1.txt':[54,59,119],
#                'selfvaild_fcn_mean803.txt':[68,118,109,131],
#                'selfvalid_RNNmean807.txt':[91,65]}
# RepalceAndEnsemble(["selfvalid_SixEnsemble2.txt"],outputFile2,replace_dict)
# "selfvalid_historyValueByday_1.0.txt","selfValid_lastValue_1.2.txt","xgb_808_1.txt","selfValid_KNN804.txt"
#,"selfValid_SVRModel808.txt","selfvalid_RNNmean807.txt","selfvaild_fcn_mean803.txt","sv_rnn_mean808.txt"
# SelfValidMAPE.processingOut(outputFile2)
# os.remove(path+outputFile)


