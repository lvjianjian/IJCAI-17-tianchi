#encoding=utf-8

import numpy as np
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.utils import np_utils
import Parameter
import pandas as pd
from cjx_predict import getCoutList
from cjx_predict import preprocessCoutList
from keras.optimizers import RMSprop
from keras.regularizers import l2, activity_l2
from keras import backend as K
from keras.layers import LSTM
from lv import toInt
from cjx_predict import scoreoneshop
from lv import removeNegetive

rnn_nb_epoch = 8

def my_loss(y_true,y_predict):
    print y_true,y_predict
    return K.mean(abs((y_predict-y_true)/(y_predict+y_true)),axis = -1)

def predictOneTrain_SRN(shopid, all_data, trainAsTest=False):
    """
    用SRN预测某一个商店
    :param shopid: 预测商店id
    :param trainAsTest: 是否使用训练集后14天作为测试集
    :return:
    """
    all_countList = getCoutList(all_data, shopId=shopid)
    seq_length = 14
    # 取出一部分做训练集
    if trainAsTest: #使用训练集后14天作为测试集的话，训练集为前面部分
        part_countList = all_countList[0:len(all_countList) - 14]
    else:
        part_countList = all_countList
    train_x, train_y = preprocessCoutList(seq_length, part_countList)

    # test_coutList = all_countList[len(all_countList) - 2*seq_length:len(all_countList)]
    # [test_x, test_y] = preprocessCoutList(seq_length, test_coutList)

    model = Sequential()
    model.add(LSTM(32, input_shape=(train_x.shape[1], train_x.shape[2]), activation="tanh")) #sigmoid
    model.add(Dense(1, activation='linear'))
    #, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)

    # 设置优化器（除了学习率外建议保持其他参数不变）
    rms=RMSprop(lr=0.03)
    # sgd=SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss=my_loss, optimizer=rms)
    print model.summary()
    model.fit(train_x, train_y, nb_epoch=rnn_nb_epoch, batch_size=1, verbose=2)

    last = train_y[len(train_y) - 1]
    last_x = train_x[train_x.shape[0] - 1]
    if trainAsTest:
        last_14_real_y = all_countList[len(all_countList) - 14:]
    #预测值
    prediction_y=[]
    for i in range(14):
        new_x = last_x[1:].copy()
        new_x = np.concatenate((new_x, [[last]]))
        new_x = np.reshape(new_x, (1, 14, 1))
        print new_x
        last = model.predict(new_x)[0][0]
        print last
        prediction_y.append(last)
        last_x = new_x[0]
    prediction_y = (removeNegetive(toInt(np.array(prediction_y)))).astype(int)
    if trainAsTest:
        print str(shopid)+',score:', scoreoneshop(prediction_y, np.array(last_14_real_y))
    return [prediction_y, shopid]

if __name__ == "__main__":
    list=[]
    for i in range(365):
        if(i%7<5):
            list.append(3)
        else:
            list.append(10)
    print list
    train_x,train_y = preprocessCoutList(14, list)

    model = Sequential()
    model.add(LSTM(32, input_shape=(train_x.shape[1], train_x.shape[2]), activation="tanh")) #sigmoid
    model.add(Dense(1, activation='linear'))
    #, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)

    # 设置优化器（除了学习率外建议保持其他参数不变）
    rms=RMSprop(lr=0.05)
    # sgd=SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=rms)
    print model.summary()
    model.fit(train_x, train_y, nb_epoch=10, batch_size=1, verbose=2)

    predict = model.predict(np.reshape(np.array([3, 3,10, 10, 3, 3, 3, 3, 3, 10, 10,3, 3, 3,]), (1, 14, 1)))
    print predict

