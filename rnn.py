from datetime import datetime
import os
import numpy as np
import tensorflow as tf
import gc
from sklearn.model_selection import KFold
import json

taskname="1"
lr=0.05
times=int(1e4)

def listmap(o,p):
    return list(map(o,p))

def loadPath():
    with open("config.json") as f:
    #with open("configSelfValid.json") as f:
        config=json.loads(f.read())
        return config["datapath"],config["sharepath"],config["rootpath"],config["selfvalidpath"],config["startdate"],config["days"]

datapath,sharepath,rootpath,selfvalidpath,startdate,days=loadPath()

def splitData(tensor,n_output,n_pred):
    print(tensor.shape)
    n_known=tensor.shape[0]-n_pred
    n_input=tensor.shape[1]-n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known+n_pred, 0: n_input]
    return (knownX,knownY,preX)

lstm_size=1
number_of_layers=2

inpath=rootpath+taskname+"\\"

tensor=np.loadtxt(inpath+"tensor_fill.csv",delimiter=',')

knownX,knownY,preX=splitData(tensor,30,days)

num_steps=knownX.shape[1]
y_steps=knownY.shape[1]
x=tf.placeholder(tf.float32, [None,num_steps])
expx=tf.expand_dims(x, -1)
yTrue=tf.placeholder(tf.float32,[None,y_steps])

def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)

with tf.variable_scope("encoder"):
    encoder = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(number_of_layers)])
    initial_state = state = encoder.zero_state(knownX.shape[0], tf.float32)
    for i in range(num_steps):
        print(i)
        if i>0:
            tf.get_variable_scope().reuse_variables()
        output, state = encoder(expx[:,i], state)

with tf.variable_scope("decoder"):
    decoder = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(number_of_layers)])
    outputs=output
    for i in range(y_steps-1):
        print(i)
        if i>0:
            tf.get_variable_scope().reuse_variables()
        outexp=tf.expand_dims(output, -1)
        output, state = decoder(outexp[:,-1], state)
        outputs=tf.concat([outputs,output],1)

lossfun=tf.reduce_mean(tf.abs(tf.subtract(outputs/yTrue,1)))

train_step=tf.train.AdamOptimizer(learning_rate=lr).minimize(lossfun)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fd={x:knownX,yTrue:knownY}
    out=sess.run(output,feed_dict=fd)
    state=sess.run(state,feed_dict=fd)
    for i in range(times):
        sess.run(train_step,feed_dict=fd)
        print(sess.run(lossfun,feed_dict=fd))
        
    