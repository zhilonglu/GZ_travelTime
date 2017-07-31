#合并几个文件成一个文件
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\result\\"
#找出三个数中最小的非0元素
# def findMinExceptZero(a):
#     tmp = sorted(a)
#     if tmp[0]==0:
#         if tmp[1]==0:
#             return tmp[2]
#         else:
#             return tmp[1]
#     else:
#         return tmp[0]
def findMinExceptZero(a):
    tmp = sorted(a)
    if tmp[0]==0:
        return tmp[1]
    else:
        return tmp[0]
def integer(file1,file2):
    result_dict_1={}
    result_dict_2={}
    # result_dict_3={}
    with open(path+file1) as f1:
        all = f1.readlines()
        for i in range(len(all)):
            values = all[i].replace("\n","").split("#")
            result_dict_1[(values[0],values[1],values[2])] = float(values[3])
    with open(path+file2) as f2:
        all = f2.readlines()
        for i in range(len(all)):
            values = all[i].replace("\n","").split("#")
            result_dict_2[(values[0],values[1],values[2])] = float(values[3])
    # with open(path+file3) as f3:
    #     all = f3.readlines()
    #     for i in range(len(all)):
    #         values = all[i].replace("\n","").split("#")
    #         result_dict_3[(values[0],values[1],values[2])] = float(values[3])
    with open(path+"submit_min_LastValueAndKNN0729.txt","w") as f4:
        for i in result_dict_1:
            f4.write("#".join(i)+"#"+str(findMinExceptZero([result_dict_1[i],result_dict_2[i]]))+"\n")
integer("submit_2.txt","submit_knn_0729_c.txt")
