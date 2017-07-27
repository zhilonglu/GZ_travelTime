<<<<<<< HEAD
from datetime import datetime
import os
import numpy as np
import tensorflow as tf
import gc
from sklearn.model_selection import KFold

def listmap(o,p):
    return list(map(o,p))

lossplots=[]
outputs=None

def splitData(tensor,n_output,n_pred):
    print(tensor.shape)
    n_known=tensor.shape[0]-n_pred
    n_input=tensor.shape[1]-n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known+n_pred, 0: n_input]
    return (knownX,knownY,preX)

def fcn(trainX,trainY,hiddennum,times,keep,modelname,cutime,validX,validY,lr):
    global outputs
    inputnum=len(trainX[0])
    outputnum=len(trainY[0])
    losscol=[]
    npx=trainX
    npy=trainY
    npx_test=validX
    npy_test=validY
    nodenums=[inputnum]+hiddennum+[outputnum]
    sess=tf.InteractiveSession()
    x=tf.placeholder(tf.float32, [None,inputnum])
    y_=tf.placeholder(tf.float32,[None,outputnum])
    keep_prob=tf.placeholder(tf.float32)
    hiddens=[]
    drops=[x]
    for i in range(len(nodenums)-1):
        if(i==len(nodenums)-2):
            Wi=tf.Variable(tf.truncated_normal([nodenums[i],nodenums[i+1]],mean=10, stddev=0.1), name="W"+str(i)+cutime)
        else:
            Wi=tf.Variable(tf.truncated_normal([nodenums[i],nodenums[i+1]],mean=0, stddev=0.1), name="W"+str(i)+cutime)
        bi= tf.Variable(tf.ones(nodenums[i+1]), name="b"+str(i)+cutime)
        if i<len(nodenums)-2:
            hiddeni = tf.nn.relu(tf.add(tf.matmul(drops[i],Wi),bi))
            hiddens.append(hiddeni)
            dropi=tf.nn.dropout(hiddeni,keep_prob)
            drops.append(dropi)
        else:
            y=tf.add(tf.matmul(drops[i],Wi),bi)
    lossfun=tf.reduce_mean(tf.abs(tf.subtract(y/y_,1)))
    train_step=tf.train.AdamOptimizer(learning_rate=lr).minimize(lossfun)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(times):
        noiseX=npx+2*np.random.random(npx.shape)-1
        noiseY=npy+2*np.random.random(npy.shape)-1
        sess.run(train_step,feed_dict={x:noiseX,y_:noiseY,keep_prob:keep})
        loss1=sess.run(lossfun,feed_dict={x:npx,y_:npy,keep_prob:1})
        loss2=sess.run(lossfun,feed_dict={x:npx_test,y_:npy_test,keep_prob:1})
        losscol.append([loss1,loss2,loss1+loss2])
    losscolnp=np.array(losscol)
    lossplots.append("'"+cutime+","+",".join(listmap(str,losscol[-1])))
    print(losscol[-1])
    # valid
    preY_valid = sess.run(y, feed_dict={x: npx_test, keep_prob: 1})
    np.savetxt(path + "preY_valid.csv", preY_valid.reshape(-1, 1), fmt="%.8f", delimiter=',')
    # test
    preY_test = sess.run(y, feed_dict={x: preX, keep_prob: 1})
    np.savetxt(path + "preY_test.csv", preY_test.reshape(-1, 1), fmt="%.8f", delimiter=',')
    if outputs is None:
        outputs=preY_test.reshape(-1, 1)
    else:
        outputs=np.c_[outputs, preY_test.reshape(-1, 1)]
    np.savetxt(path+"losscol.csv",losscolnp,fmt="%.8f",delimiter=',')
    del sess
    del hiddens
    del drops
    del losscolnp
    del losscol
    gc.collect()

datapath = 'C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\'
allTask=[]
with open(datapath+"gy_contest_link_info.txt") as f:
    f.readline()
    all = f.readlines()
    for i in range(len(all)):
        allTask.append(all[i].split(";")[0])
for taskname in allTask:
    rootpath=datapath+"tensorData9\\"
    inpath=rootpath+taskname+"\\"
    cutime=datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
    if not os.path.exists(inpath):
        continue
    tensor=np.loadtxt(inpath+"tensor_fill.csv",delimiter=',')
    knownX,knownY,preX=splitData(tensor,30,30)
    alltimes=0
    for i in range(5):
        kf = KFold(n_splits=int(knownX.shape[0]))
        for train_index, valid_index in kf.split(knownX):
            print("TRAIN:", train_index, "VALID:", valid_index)
            cutime=datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
            path=inpath+cutime+"\\"
            os.makedirs(path)
            print(path)
            trainX, validX = knownX[train_index], knownX[valid_index]
            trainY, validY = knownY[train_index], knownY[valid_index]
            print(trainX.shape,trainY.shape,validX.shape,validY.shape,preX.shape)
            np.savetxt(path+"validYtrue.csv",validY.reshape(-1, 1), fmt="%.8f",delimiter=',')
            lossi=fcn(trainX, trainY, [60,60,30], int(5e4), 0.9, taskname, cutime, validX, validY, 3e-4)
            with open(path +"lossplots"+datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")+".csv","w") as f:
                f.write("\n".join(lossplots))
            if(alltimes>=9):
                break
            else:
                alltimes=alltimes+1
        if(alltimes>=9):
            break
    cutime=datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
    np.savetxt(inpath + "outputs.csv", outputs, fmt="%.8f", delimiter=',')
    np.savetxt(inpath+ "outputsmedian.csv", np.median(outputs,1), fmt="%.8f", delimiter=',')
=======
from datetime import datetime
import os
import numpy as np
import tensorflow as tf
import gc
from sklearn.model_selection import KFold

def listmap(o,p):
    return list(map(o,p))

lossplots=[]
outputs=None

def splitData(tensor,n_output,n_pred):
    print(tensor.shape)
    n_known=tensor.shape[0]-n_pred
    n_input=tensor.shape[1]-n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known+n_pred, 0: n_input]
    return (knownX,knownY,preX)

def fcn(trainX,trainY,hiddennum,times,keep,modelname,cutime,validX,validY,lr):
    global outputs
    inputnum=len(trainX[0])
    outputnum=len(trainY[0])
    losscol=[]
    npx=trainX
    npy=trainY
    npx_test=validX
    npy_test=validY
    nodenums=[inputnum]+hiddennum+[outputnum]
    sess=tf.InteractiveSession()
    x=tf.placeholder(tf.float32, [None,inputnum])
    y_=tf.placeholder(tf.float32,[None,outputnum])
    keep_prob=tf.placeholder(tf.float32)
    hiddens=[]
    drops=[x]
    for i in range(len(nodenums)-1):
        if(i==len(nodenums)-2):
            Wi=tf.Variable(tf.truncated_normal([nodenums[i],nodenums[i+1]],mean=10, stddev=0.1), name="W"+str(i)+cutime)
        else:
            Wi=tf.Variable(tf.truncated_normal([nodenums[i],nodenums[i+1]],mean=0, stddev=0.1), name="W"+str(i)+cutime)
        bi= tf.Variable(tf.ones(nodenums[i+1]), name="b"+str(i)+cutime)
        if i<len(nodenums)-2:
            hiddeni = tf.nn.relu(tf.add(tf.matmul(drops[i],Wi),bi))
            hiddens.append(hiddeni)
            dropi=tf.nn.dropout(hiddeni,keep_prob)
            drops.append(dropi)
        else:
            y=tf.add(tf.matmul(drops[i],Wi),bi)
    lossfun=tf.reduce_mean(tf.abs(tf.subtract(y/y_,1)))
    train_step=tf.train.AdamOptimizer(learning_rate=lr).minimize(lossfun)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(times):
        noiseX=npx+2*np.random.random(npx.shape)-1
        noiseY=npy+2*np.random.random(npy.shape)-1
        sess.run(train_step,feed_dict={x:noiseX,y_:noiseY,keep_prob:keep})
        loss1=sess.run(lossfun,feed_dict={x:npx,y_:npy,keep_prob:1})
        loss2=sess.run(lossfun,feed_dict={x:npx_test,y_:npy_test,keep_prob:1})
        losscol.append([loss1,loss2,loss1+loss2])
    losscolnp=np.array(losscol)
    lossplots.append("'"+cutime+","+",".join(listmap(str,losscol[-1])))
    print(losscol[-1])
    # valid
    preY_valid = sess.run(y, feed_dict={x: npx_test, keep_prob: 1})
    np.savetxt(path + "preY_valid.csv", preY_valid.reshape(-1, 1), fmt="%.8f", delimiter=',')
    # test
    preY_test = sess.run(y, feed_dict={x: preX, keep_prob: 1})
    np.savetxt(path + "preY_test.csv", preY_test.reshape(-1, 1), fmt="%.8f", delimiter=',')
    if outputs is None:
        outputs=preY_test.reshape(-1, 1)
    else:
        outputs=np.c_[outputs, preY_test.reshape(-1, 1)]
    np.savetxt(path+"losscol.csv",losscolnp,fmt="%.8f",delimiter=',')
    del sess
    del hiddens
    del drops
    del losscolnp
    del losscol
    gc.collect()

datapath = 'C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\'
allTask=[]
with open(datapath+"gy_contest_link_info.txt") as f:
    f.readline()
    all = f.readlines()
    for i in range(len(all)):
        allTask.append(all[i].split(";")[0])
for taskname in allTask:
    rootpath=datapath+"tensorData9\\"
    inpath=rootpath+taskname+"\\"
    cutime=datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
    if not os.path.exists(inpath):
        continue
    tensor=np.loadtxt(inpath+"tensor_fill.csv",delimiter=',')
    knownX,knownY,preX=splitData(tensor,30,30)
    alltimes=0
    for i in range(5):
        kf = KFold(n_splits=int(knownX.shape[0]))
        for train_index, valid_index in kf.split(knownX):
            print("TRAIN:", train_index, "VALID:", valid_index)
            cutime=datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
            path=inpath+cutime+"\\"
            os.makedirs(path)
            print(path)
            trainX, validX = knownX[train_index], knownX[valid_index]
            trainY, validY = knownY[train_index], knownY[valid_index]
            print(trainX.shape,trainY.shape,validX.shape,validY.shape,preX.shape)
            np.savetxt(path+"validYtrue.csv",validY.reshape(-1, 1), fmt="%.8f",delimiter=',')
            lossi=fcn(trainX, trainY, [60,60,30], int(5e4), 0.9, taskname, cutime, validX, validY, 3e-4)
            with open(path +"lossplots"+datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")+".csv","w") as f:
                f.write("\n".join(lossplots))
            if(alltimes>=9):
                break
            else:
                alltimes=alltimes+1
        if(alltimes>=9):
            break
    cutime=datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
    np.savetxt(inpath + "outputs.csv", outputs, fmt="%.8f", delimiter=',')
    np.savetxt(inpath+ "outputsmedian.csv", np.median(outputs,1), fmt="%.8f", delimiter=',')
>>>>>>> origin/master
    np.savetxt(inpath+ "outputsmean.csv", np.mean(outputs,1), fmt="%.8f", delimiter=',')