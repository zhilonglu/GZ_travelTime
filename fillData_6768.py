
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\tensorData5\\"
for idx in range(1,133):
    with open(path+str(idx)+"\\tensor.csv") as f:
        with open(path+str(idx)+"\\tensor_fill2.csv","w") as f2:
            data = [[0 for col in range(90)] for row in range(122)]
            all = f.readlines()
            sum = 0
            cnt = 0
            for l in range(len(all)):
                values = list(map(float,all[l].replace("\n","").split(",")))
                for v in range(len(values)):
                    if values[v] >0:
                        data[l][v] = values[v]
                        sum += values[v]
                        cnt += 1
            for i in range(122):
                for j in range(90):
                    if data[i][j]==0:
                        data[i][j] = float("%.3f"%(sum/cnt))
                f2.write(",".join(map(str,data[i]))+"\n")