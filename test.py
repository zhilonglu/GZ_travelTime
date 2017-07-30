import numpy as np
import os
def splitData(tensor,n_output,n_pred):
    print(tensor.shape)
    n_known=tensor.shape[0]-n_pred
    n_input=tensor.shape[1]-n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known+n_pred, 0: n_input]
    return (knownX,knownY,preX)
inpath="C:\\Users\\NLSDE\\Desktop\\kdd\\最后提交\\data_10\\1_0pm\\"
tensor=np.loadtxt(inpath+"tensor_new.csv",delimiter=',')
# knownX,knownY,preX=splitData(tensor,6,7)
# print(knownX)
# print(knownY)
# print(preX)
# dict_data={}
# for i in range(2):
#     dict_data[i]=[]
#     for j in range(3):
#         for k in range(3):
#             dict_data[i].append((j,k))
# print(dict_data)
# dt = {}
# dt[0]=[(21,1),(12,3)]
# dt[1]=[(23,1),(12,3),(14,5)]
# sortDT={}
# for i in dt:
#     sorted(dt[i])
#     # sortDT[i]=tempList
# print(dt)

a=[211,2312,12312,12,243,43,211]
c={}
for i in range(len(a)):
    if a[i] not in c.keys():
        c[a[i]]=len(c)
# print(c)

a=[1,2,3,4,5]
# for i in range(len(a)-2,0,-1):
#     # print(a[i])

# import pandas
path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\tensorData\\"
# data = pandas.read_csv(path+"test.csv")
# # print(data)
# print(data.shape)
# print(data.loc[5][9])
# import pandas as pd
# import numpy as np
#
# df = pd.DataFrame({'total_bill': [16.99, 10.34, 23.68, 23.68, 24.59],
#                    'tip': [1.01, 1.66, 3.50, 3.31, 3.61],
#                    'sex': ['Female', 'Male', 'Male', 'Male', 'Female']})
# print(df)
# # data type of columns
# print(df.dtypes)
# # indexes
# print(df.index)
# # return pandas.Index
# print(df.columns)
# # each row, return array[array]
# print(df.values)
# arima_data={}
# for i in range(30):
#     arima_data[i] = []
# files = os.listdir(path)
# files_id = os.listdir(path + "3377906280028510514")
# for file in files_id:
#     if file == "tensor_fill.csv":
#         with open(path + "3377906280028510514"+ "\\" + file) as f2:
#             all = f2.readlines()
#             for j in range(0,len(all)-30):
#                 tempdata = list(map(float, all[j].replace("\n", "").split(",")))
#                 for k in range(60, 90):
#                     arima_data[k - 60].append(tempdata[k])
#             print(len(arima_data[0]))

# a={'12':[(1,2),(2,3),(3,4)],
#    '13':[(1,2),(2,3),(3,4)],
#    '14':[(1,2),(2,3),(3,4)]}
# for i in a:
#     print(a[i])
#     tuple_list = []
#     for j in a[i]:
#         tuple_list.append(j[1])
#     print(tuple_list)

# import matplotlib.pyplot as plt
# #
# x = range(10)
# y1 = [elem*2 for elem in x]
# plt.plot(x, y1,color="blue", linewidth=2.5, linestyle="-", label="cosine")
#
# y2 = [elem**2 for elem in x]
# plt.plot(x, y2,color="red",  linewidth=2.5, linestyle="-", label="sine")
# plt.legend(loc='upper left')
# plt.show()


a=0
b=4
# if 0 in [a,b]:
#     print(max(a,b))
# else:
#     print(min(a,b))
# c = max(a,b) if 0 in [a,b] else min(a,b)
# print(c)

# y1 = [30,30,40,40,50,50]
# y2 = [40,40,40,40,40,40]
# y3 = [50,50,50,50,50,50]
# y = list(map(lambda x:x[0]+x[1]+x[2],zip(y1,y2,y3)))
# r = [x/sum(y) for x in y]
# print(r)
# print(y)

X = [[0], [1], [2], [3]]
y = [0, 1, 4, 9]
# from sklearn.neighbors import KNeighborsRegressor
# neigh = KNeighborsRegressor(n_neighbors=2)
# neigh.fit(X, y)
# pre_y = neigh.predict([[1.5],[2],[3]])
# print(neigh.predict([[1.5],[2],[3]]))

# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0)
# model.fit(X,y)
# print(model.predict([[1.5],[2],[3]]))

# y = [0.3*x for x in y]
# y2 = 3*y
# print(y)
# print(y2)

# a="sfsdfsdfsdf:sd"
# b = a.encode('utf-8').strip()
# c = str(b, encoding="utf-8")
# print(c)

# import numpy as np
# import matplotlib.pyplot as plt
# # plt.figure(1)#创建图表1
# plt.figure(2)#创建图表2
# ax1=plt.subplot(211)#在图表2中创建子图1
# ax2=plt.subplot(212)#在图表2中创建子图2
# x=np.linspace(0,3,100)
# for i in range(5):
#     # plt.figure(1)
#     # plt.plot(x,np.exp(i*x/3))
#     plt.sca(ax1)
#     plt.plot(x,np.sin(i*x))
#     plt.sca(ax2)
#     plt.plot(x,np.cos(i*x))
# plt.show()

# test1 = np.array([[5, 10, 15],
#             [20, 25, 30],
#             [35, 40, 45]])
# x = test1.sum(axis=0).reshape(1,-1)
# print(x)
# y = x/x.sum()
# print(y[0])

a = [1,2,0]
b = [2,3,1]
c = [0,1,0]
def findMinExceptZero(a):
    tmp = sorted(a)
    if tmp[0]==0:
        if tmp[1]==0:
            return tmp[2]
        else:
            return tmp[1]
    else:
        return tmp[0]
print(findMinExceptZero(c))
