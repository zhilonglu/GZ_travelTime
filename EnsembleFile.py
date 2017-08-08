#合并几个文件成一个文件
import SelfValidMAPE
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\result\\"
#进行selfValid时启用的代码
# path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\selfValid\\"
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
                result_dict[idx][(values[0],values[1],values[2])] = float(values[3])
    with open(path+outputfile,"w") as f4:
        for i in result_dict[0]:
            tempList = []
            for j in range(len(files)):
                tempList.append(result_dict[j][i])
            f4.write("#".join(i)+"#"+str(findSecondMinExceptZero(tempList))+"\n")
outputfile = "submit_SixEnsemble808.txt"
integer(["submit_RNN_mean807.txt","submit_lastValue1.2.txt","knn_804.txt","submit_fcn_mean5.txt"
         ,"SVR_rbfModel0802.txt","submit_historyValueByday1.0.txt"],outputfile)
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
# outputFile = "selfvalid_SixEnsemble.txt"
# integer(["selfvalid_RNNmean807.txt","selfValid_SVRModel.txt","selfValid_KNN804.txt","selfvaild_fcn_mean803.txt",
#         "selfvalid_historyValueByday_1.0.txt","selfValid_lastValue_1.2.txt"],outputFile)
# SelfValidMAPE.processingOut(outputFile)
