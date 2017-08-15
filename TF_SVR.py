from datetime import datetime
import os
import numpy as np
import tensorflow as tf
import gc
from sklearn.model_selection import KFold
from sklearn.svm import SVR
import json
import threading


def listmap(o, p):
    return list(map(o, p))

def loadPath():
    with open("config.json") as f:
        # with open("configSelfValid.json") as f:
        config = json.loads(f.read())
        return config["datapath"], config["sharepath"], config["rootpath"], config["tensorfile"], config["startdate"], \
               config["days"]

datapath, sharepath, rootpath, tensorfile, startdate, days = loadPath()

def build_SVR(hiddennum, inputnum, outputnum):
    x = tf.placeholder(tf.float32, [None, inputnum])
    yTrue = tf.placeholder(tf.float32, [None, outputnum])
    keep_prob = tf.placeholder(tf.float32)
    nodenums = [inputnum] + hiddennum + [outputnum]
    hiddens = []
    drops = [x]
    for i in range(len(nodenums) - 1):
        if (i == len(nodenums) - 2):
            Wi = tf.Variable(tf.truncated_normal([nodenums[i], nodenums[i + 1]], mean=10, stddev=0.1))
        else:
            Wi = tf.Variable(tf.truncated_normal([nodenums[i], nodenums[i + 1]], mean=0, stddev=0.1))
        bi = tf.Variable(tf.ones(nodenums[i + 1]))
        if i < len(nodenums) - 2:
            hiddeni = tf.nn.relu(tf.add(tf.matmul(drops[i], Wi), bi))
            hiddens.append(hiddeni)
            dropi = tf.nn.dropout(hiddeni, keep_prob)
            drops.append(dropi)
        else:
            y = tf.add(tf.matmul(drops[i], Wi), bi)
    lossfun = tf.reduce_mean(tf.abs(tf.subtract(y / yTrue, 1)))
    return (x, y, yTrue, keep_prob, lossfun)

def splitData(tensor, n_output, n_pred):
    n_known = tensor.shape[0] - n_pred
    n_input = tensor.shape[1] - n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known + n_pred, 0: n_input]
    return (knownX, knownY, preX)

def fcn(taskname, path, tensors, _fcn, times, keep, lr, outputs):
    npx, npx_test, npy, npy_test, preX = tensors
    x, y, yTrue, keep_prob, lossfun = _fcn
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(lossfun)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(times):
            sess.run(train_step, feed_dict={x: npx, yTrue: npy, keep_prob: keep})
            loss1 = sess.run(lossfun, feed_dict={x: npx, yTrue: npy, keep_prob: 1})
            loss2 = sess.run(lossfun, feed_dict={x: npx_test, yTrue: npy_test, keep_prob: 1})
        print(taskname, ":", loss1, loss2)
        preY_valid = sess.run(y, feed_dict={x: npx_test, keep_prob: 1})
        np.savetxt(path + "preY_valid.csv", preY_valid.reshape(-1, 1), fmt="%.8f", delimiter=',')
        # test
        preY_test = sess.run(y, feed_dict={x: preX, keep_prob: 1})
        np.savetxt(path + "preY_test.csv", preY_test.reshape(-1, 1), fmt="%.8f", delimiter=',')
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
    inputnum = knownX.shape[1]
    outputnum = knownY.shape[1]
    _fcn = build_SVR([60, 60, 30], inputnum, outputnum)
    for i in range(2):
        kf = KFold(n_splits=10)
        for train_index, valid_index in kf.split(knownX):
            print("taskname:", taskname, "i:", i)
            cutime = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
            path = inpath + cutime + "\\"
            os.makedirs(path)
            tensors = knownX[train_index], knownX[valid_index], knownY[train_index], knownY[valid_index], preX
            outputs = fcn(taskname, path, tensors, _fcn, int(3e4), 0.88, 3e-4, outputs)
            np.savetxt(path + "validYtrue.csv", knownY[valid_index].reshape(-1, 1), fmt="%.8f", delimiter=',')
    sharetaskpath = sharepath + taskname + "\\"
    if not os.path.exists(sharetaskpath):
        os.makedirs(sharetaskpath)
    np.savetxt(sharetaskpath + "outputsmedian.csv", np.median(outputs, 1), fmt="%.8f", delimiter=',')
    np.savetxt(sharetaskpath + "outputsmean.csv", np.mean(outputs, 1), fmt="%.8f", delimiter=',')

allTask = listmap(str, list(range(1, 2)))
print(allTask)

while (len(allTask) > 0):
    if (len(allTask) > 1):
        cutasks = allTask[0:1]
        allTask = allTask[1::]
    else:
        cutasks = allTask
        allTask = []
    threads = []
    for taskname in cutasks:
        threads.append(threading.Thread(target=runtask, args=(taskname,)))

    for t in threads:
        t.setDaemon(True)
        t.start()

    for t in threads:
        t.join()