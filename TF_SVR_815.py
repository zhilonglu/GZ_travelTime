import os
import numpy as np
import tensorflow as tf
import gc
from sklearn.model_selection import KFold
import json

def listmap(o, p):
    return list(map(o, p))

def loadPath():
    #with open("config.json") as f:
    with open("consvfor811.json") as f:
    #with open("configextended.json") as f:
    #with open("configSelfValid.json") as f:
    #with open("configextendedsv.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"],config["tensorfile"],config["startdate"],config["days"]

datapath,sharepath,rootpath,tensorfile,startdate,days=loadPath()

def splitData(tensor, n_output, n_pred):
    n_known = tensor.shape[0] - n_pred
    n_input = tensor.shape[1] - n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known + n_pred, 0: n_input]
    return (knownX, knownY, preX)

def build_SVR():
    x = tf.placeholder(tf.float32,shape= [None,60])
    W = tf.Variable(tf.ones([60,1]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,W)
    b = tf.Variable(10*tf.ones([30]))
    y = (tf.matmul(x,W)+b)
    y_ = tf.placeholder(tf.float32,shape=[None,30])
    
    lossfun = tf.reduce_mean(tf.abs(tf.subtract(y / y_, 1)))
    return (x, y, y_, lossfun)

def TF_process(taskname, tensors, _fcn, times, lr, outputs):
    npx, npx_test, npy, npy_test, preX = tensors
    x, y, y_,lossfun = _fcn
    lossfun2 = tf.reduce_mean(tf.abs(tf.subtract(y , y_)))
    regularizer=tf.contrib.layers.l1_regularizer(100.0)
    l1_loss=tf.contrib.layers.apply_regularization(regularizer)
    regularizer2=tf.contrib.layers.l2_regularizer(100.0)
    l2_loss=tf.contrib.layers.apply_regularization(regularizer2)
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(lossfun+lossfun2+l1_loss+l2_loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(times):
            sess.run(train_step, feed_dict={x: npx, y_: npy})
            loss1 = sess.run(lossfun, feed_dict={x: npx, y_: npy})
            loss2 = sess.run(lossfun, feed_dict={x: npx_test, y_: npy_test})
        print(taskname, ":", loss1, loss2)
        preY_test = sess.run(y, feed_dict={x: preX})
        if outputs is None:
            outputs = preY_test.reshape(-1, 1)
        else:
            outputs = np.c_[outputs, preY_test.reshape(-1, 1)]
    del sess
    gc.collect()
    return outputs

if not os.path.exists(sharepath):
    os.makedirs(sharepath)

def runtask(taskname):
    outputs = None
    inpath = rootpath + taskname + "\\"
    tensor = np.loadtxt(inpath + tensorfile, delimiter=',')
    knownX, knownY, preX = splitData(tensor, 30, days)
    _fcn = build_SVR()
    for i in range(1):
        kf = KFold(n_splits=4,shuffle=True)
        for train_index, valid_index in kf.split(knownX):
            print("taskname:", taskname, "i:", i)
            trainX=knownX[train_index]
            xmean=np.mean(knownX,axis=0)
            xsigma=np.std(knownX,axis=0)
            phiTrainX=np.exp(-1*np.square(trainX-xmean)/(2*np.square(xsigma)))
            validX=knownX[valid_index]
            phiValidX=np.exp(-1*np.square(validX-xmean)/(2*np.square(xsigma)))
            phiPreX=np.exp(-1*np.square(preX-xmean)/(2*np.square(xsigma)))
            tensors = phiTrainX, phiValidX, knownY[train_index], knownY[valid_index], phiPreX
            outputs = TF_process(taskname, tensors, _fcn, int(1e4),0.0005, outputs)
    sharetaskpath = sharepath + taskname + "\\"
    if not os.path.exists(sharetaskpath):
        os.makedirs(sharetaskpath)
    np.savetxt(sharetaskpath + "outputsmedian.csv", np.median(outputs, 1), fmt="%.8f", delimiter=',')
    np.savetxt(sharetaskpath + "outputsmean.csv", np.mean(outputs, 1), fmt="%.8f", delimiter=',')

allTask = listmap(str, list(range(1, 2)))
print(allTask)
for i in allTask:
    runtask(i)