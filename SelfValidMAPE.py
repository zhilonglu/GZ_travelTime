'''
Created on 2017-7-31

@author: Administrator
'''
import numpy as np
import os
import datetime
import json
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

def loadPath():
    with open("config.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"],config["selfvalidpath"]

datapath,sharepath,rootpath,selfvalidpath=loadPath()

toBeValided=datapath+"selfValid\\selfValid_RF300Model.txt"

TrueY=datapath+"selfValid\\selfValid_TrueY.txt"

TrueYFill=datapath+"selfValid\\selfValid_TrueYFill.txt"

def mape(toBeValided,TrueY):
    YDict={}
    with open(TrueY) as f:
        f_all=f.read()
        lines=f_all.split("\n")
        for line in lines:
            ls=line.split("#")
            if(len(ls)==4):
                YDict[(ls[0],ls[1],ls[2])]=float(ls[3])
    
    with open(toBeValided) as f:
        sumloss=0
        notzero=0
        f_all=f.read()
        lines=f_all.split("\n")
        for line in lines:
            ls=line.split("#")
            if(len(ls)==4):
                prey=float(ls[3])
                truey=YDict[(ls[0],ls[1],ls[2])]
                if truey==0:
                    continue
                else:
                    notzero+=1
                    sumloss+=abs(prey-truey)/truey
    return sumloss/notzero

print(mape(toBeValided,TrueY))
print(mape(toBeValided,TrueYFill))

