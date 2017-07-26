#合并几个文件成一个文件
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"
def integer(file1,file2):
    result_dict_1={}
    result_dict_2={}
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
    with open(path+"submit_3.txt","w") as f3:
        for i in result_dict_1:
            f3.write("#".join(i)+"#"+str(max(result_dict_1[i],result_dict_2[i]))+"\n")
integer("submit_1.txt","submit_2.txt")