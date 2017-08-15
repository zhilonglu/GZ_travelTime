'''
Created on 2017-7-31

@author: Administrator
'''
from datetime import datetime
import os
import numpy as np
import tensorflow as tf
import gc
from sklearn.model_selection import KFold
import json
import threading

def listmap(o,p):
    return list(map(o,p))

def loadPath():
    # with open("config.json") as f:
    # with open("configfor811.json") as f:
    with open("configSelfValid.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"],config["startdate"],config["days"]

datapath,sharepath,rootpath,startdate,days=loadPath()

print(datapath,sharepath,rootpath,startdate,days)

def splitData(tensor,n_output,n_pred):
    n_known=tensor.shape[0]-n_pred
    n_input=tensor.shape[1]-n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known+n_pred, 0: n_input]
    return (knownX,knownY,preX)

def lstm_cell(lstm_size):
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)

def seq2seq(inputs, lstm_size, number_of_layers, batch_size, y_steps):
    num_steps=inputs.shape[1]
    expx=tf.expand_dims(inputs, -1)
    with tf.variable_scope("encoder"):
        encoder = tf.contrib.rnn.MultiRNNCell(
            [lstm_cell(lstm_size) for _ in range(number_of_layers)])
        initial_state = state = encoder.zero_state(batch_size, tf.float32)
        for i in range(num_steps):
            if i>0:
                tf.get_variable_scope().reuse_variables()
            output, state = encoder(expx[:,i], state)
    with tf.variable_scope("decoder"):
        decoder = tf.contrib.rnn.MultiRNNCell(
            [lstm_cell(lstm_size) for _ in range(number_of_layers)])
        outputs=output
        for i in range(y_steps-1):
            if i>0:
                tf.get_variable_scope().reuse_variables()
            outexp=tf.expand_dims(output, -1)
            output, state = decoder(outexp[:,-1], state)
            outputs=tf.concat([outputs,output],1)
    with tf.variable_scope("linearlayer"):
        W=tf.Variable(tf.truncated_normal([y_steps],mean=10, stddev=0.1))
        b=tf.Variable(tf.truncated_normal([y_steps],mean=0, stddev=0.1))
        y=W*outputs+b
    return y

def rnn(taskname,tensors,s2s,outputs):
    trainX, validX, trainY, validY, preX = tensors
    x,y,yTrue,batch_size=s2s
    lr=0.05
    times=int(1e3)
    y_steps=trainY.shape[1]
    lossfun=tf.reduce_mean(tf.abs(tf.subtract(y/yTrue,1)))
    train_step=tf.train.AdamOptimizer(learning_rate=lr).minimize(lossfun)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        fd={x:trainX,yTrue:trainY,batch_size:trainX.shape[0]}
        fd_valid={x:validX,yTrue:validY,batch_size:validX.shape[0]}
        fd_test={x:preX,batch_size:preX.shape[0]}
        for i in range(times):
            sess.run(train_step,feed_dict=fd)
        print(taskname,":",sess.run(lossfun,feed_dict=fd),sess.run(lossfun,feed_dict=fd_valid))
        preY_test = sess.run(y, feed_dict=fd_test)
        if outputs is None:
            outputs=preY_test.reshape(-1, 1)
        else:
            outputs=np.c_[outputs, preY_test.reshape(-1, 1)]
    del sess
    gc.collect()
    return outputs

def runtask(taskname):
    lstm_size=1
    number_of_layers=2
    num_steps=60
    y_steps=30
    batch_size=tf.placeholder(tf.int32)
    x=tf.placeholder(tf.float32, [None,num_steps])
    yTrue=tf.placeholder(tf.float32,[None,y_steps])
    y=seq2seq(x,lstm_size, number_of_layers, batch_size, y_steps)
    s2s=(x,y,yTrue,batch_size)
    print(taskname)
    outputs=None
    inpath=rootpath+taskname+"\\"
    tensor=np.loadtxt(inpath+"tensor_fill.csv",delimiter=',')
    knownX,knownY,preX=splitData(tensor,30,days)
    for i in range(1):
        kf = KFold(n_splits=4,shuffle=True)
        for train_index, valid_index in kf.split(knownX):
            tensors = knownX[train_index], knownX[valid_index],knownY[train_index], knownY[valid_index], preX
            outputs=rnn(taskname,tensors,s2s,outputs)
    sharetaskpath=sharepath+taskname+"\\"
    if not os.path.exists(sharetaskpath):
        os.makedirs(sharetaskpath)
    np.savetxt(sharetaskpath+ "outputsmedian.csv", np.median(outputs,1), fmt="%.8f", delimiter=',')
    np.savetxt(sharetaskpath+ "outputsmean.csv", np.mean(outputs,1), fmt="%.8f", delimiter=',')

allTask=listmap(str, list(range(111,121)));

while(len(allTask)>0):
    if(len(allTask)>1):
        cutasks=allTask[0:1]
        allTask=allTask[1::]
    else:
        cutasks=allTask
        allTask=[]
    threads = []
    for taskname in cutasks:
        threads.append(threading.Thread(target=runtask,args=(taskname,)))
    
    for t in threads:
        t.setDaemon(True)
        t.start()
    
    for t in threads:
        t.join()