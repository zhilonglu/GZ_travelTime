import json

def loadPath():
    with open("config.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"],config["selfvalidpath"]

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
def processingOut(filename):
    datapath, sharepath, rootpath, selfvalidpath = loadPath()
    toBeValided = datapath + "selfValid\\"+filename
    TrueY = datapath + "selfValid\\selfValid_TrueY.txt"
    TrueYFill = datapath + "selfValid\\selfValid_TrueYFill.txt"
    print(filename+" TrueY MAPE: %f"%mape(toBeValided,TrueY))
    print(filename+" TrueYFill MAPE: %f"%mape(toBeValided,TrueYFill))
if __name__ == '__main__':
        processingOut("nobugfcn_selfvalid_median.txt")

