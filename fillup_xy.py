import os

path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\tensorData2\\"
files = os.listdir(path)
for i in files:
    files_id = os.listdir(path+i)
    for file in files_id:
        if file=="tensor.csv":
            with open(path+i+"\\"+"tensor_fill.csv","w") as f1:
                with open(path+i+"\\"+file) as f2:
                    all = f2.readlines()
                    for j in range(len(all)):
                        str_list = list(map(float,all[j].replace("\n","").split(",")))
                        for k in range(1,len(str_list)-1):
                            if str_list[k]==0:
                                str_list[k] = max(str_list[k-1],str_list[k+1])#相邻两个元素取最大
                        if str_list[len(str_list)-1]==0:
                            str_list[len(str_list) - 1] = str_list[len(str_list) - 2]
                        for m in range(len(str_list)-2,0,-1):
                            if str_list[m]==0:
                                str_list[m] = max(str_list[m-1],str_list[m+1])#相邻两个元素取最大
                        if str_list[0]==0:
                            str_list[0] = str_list[1]
                        f1.write(",".join(list(map(str,str_list)))+"\n")
