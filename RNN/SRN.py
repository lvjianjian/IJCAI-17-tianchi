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
from lv import toInt,removeNegetive
from sklearn.preprocessing import MinMaxScaler
from FeatureExtractor import *
import datetime
from cjx_predict import scoreoneshop, score
import threading
import Queue
import gc
from sys import getrefcount

def my_loss(y_true,y_predict):
    K.clip()
    return K.mean(abs((y_predict-y_true)/(y_predict+y_true)), axis = -1)

def predictOneShop_LSTM(shopid, all_data, trainAsTest=False):
    """
    用SRN预测某一个商店
    :param shopid: 预测商店id
    :param trainAsTest: 是否使用训练集后14天作为测试集
    :return:
    """
    part_data = all_data[all_data.shopid == shopid]
    last_14_real_y = None
    # 取出一部分做训练集
    if trainAsTest: #使用训练集后14天作为测试集的话，训练集为前面部分
        last_14_real_y = part_data[len(part_data) - 14:]["count"].values
        part_data = part_data[0:len(part_data) - 14]
    # print last_14_real_y
    verbose = 2
    rnn_nb_epoch = 5
    skipNum = 0
    backNum = 14
    learnrate = 0.01
    sameday = extractBackDay(part_data, backNum, skipNum, nan_method_sameday_mean)
    count = extractCount(part_data, skipNum)
    train_x = getOneWeekdayFomExtractedData(sameday)
    train_y = getOneWeekdayFomExtractedData(count)
    '''将t标准化'''
    x_scaler = MinMaxScaler().fit(train_x)
    y_scaler = MinMaxScaler().fit(train_y)
    train_x = x_scaler.transform(train_x)
    train_y = y_scaler.transform(train_y)
    '''标准化结束'''
    train_x = train_x.reshape((train_x.shape[0],
                               train_x.shape[1], 1))
    model = Sequential()
    # print getrefcount(model)
    model.add(LSTM(32, input_shape=(train_x.shape[1], train_x.shape[2]), activation="tanh")) #sigmoid
    # print getrefcount(model)
    model.add(Dense(1, activation='linear'))
    #, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)
    # print getrefcount(model)
    # 设置优化器（除了学习率外建议保持其他参数不变）
    rms = RMSprop(lr=learnrate)
    # sgd=SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=rms)
    print model.summary()
    # print getrefcount(model)
    # print model.summary()
    model.fit(train_x, train_y, nb_epoch=rnn_nb_epoch, batch_size=1, verbose=verbose)
    # print model.get_weights()
    # part_counts = []
    # for i in range(7):
    #     weekday = i + 1
    #     part_count = getOneWeekdayFomExtractedData(count, weekday)
    #     part_counts.append(part_count)

    # print getrefcount(model)
    format = "%Y-%m-%d"
    if trainAsTest:
        startTime = datetime.datetime.strptime("2016-10-18", format)
    else:
        startTime = datetime.datetime.strptime("2016-11-1", format)
    timedelta = datetime.timedelta(1)
    preficts = []
    for i in range(14):
        currentTime = startTime + timedelta * i
        strftime = currentTime.strftime(format)
        # index = getWeekday(strftime) - 1
        # part_count = part_counts[index]
        #取前{backNum}周同一天的值为特征进行预测
        x=[]
        for j in range(backNum):
            x.append(train_y[len(train_y) - (j+1)][0])
        x = np.array(x)
        # print x
        x = x.reshape(1, backNum, 1)
        predict = model.predict(x)
        preficts.append(y_scaler.inverse_transform(predict)[0][0])
        train_y = np.append(train_y,predict).reshape((train_y.shape[0] + 1,1))
        # preficts.append(predict)
        # part_counts[index] = np.append(part_count, predict).reshape((part_count.shape[0] + 1, 1))
    preficts = (removeNegetive(toInt(np.array(preficts)))).astype(int)
    # preficts = np.array(preficts)
    if trainAsTest:
        last_14_real_y = (removeNegetive(toInt(np.array(last_14_real_y)))).astype(int)
        # print preficts,last_14_real_y
        print str(shopid)+',score:', scoreoneshop(preficts, last_14_real_y)
    return [preficts, last_14_real_y]


def predict_all(all_data, save_filename, trainAsTest=False, region=None, predict_function = None):
    """
    预测所有商店后14天的值
    :param all_data:
    :param save_filename: trainAsTest为true,则存储predict+real的值,前面14列为predict,后面14列为real,否则只有前面14列
    :param trainAsTest: 是否把训练集后14天当作测试集
    :param region: shopid区域，list,[startid,endid]
    :param predict_function: 预测单一商店的函数
    :return:
    """
    if predict_function is None:
        raise Exception("predict_function is none")
        return
    if region is None:
        startid = 1
        endid = 2000
    else:
        startid = region[0]
        endid = region[1]
    size = endid - startid + 1
    if not trainAsTest:
        result = np.zeros((size, 14))
    else:
        result = np.zeros((size, 28))
    real = np.ndarray(0)
    for i in range(startid, endid + 1, 1):
        shopid = i
        print "shopid:", shopid
        predict, real_14 = predict_function(shopid, all_data, trainAsTest)
        # scoreoneshop1 = scoreoneshop(predict, real_14)
        # predict = real_14 = np.arange(14)
        if trainAsTest:
            real = np.append(real, real_14)
            predict = np.append(predict,real_14)
        result[i-startid] = predict
        gc.collect()


    result = pd.DataFrame(result.astype(np.int))
    result.insert(0, "id", value=range(startid, endid + 1, 1))
    # print result
    result = result.values
    if(save_filename is not None):
        np.savetxt(save_filename, result, delimiter=",", fmt='%d')
    else:
        print result
    return result


def predict_all_getbest(all_data, save_filename, trainAsTest=False, region=None, predict_function = None, epoch_n = 1):
    """
    预测所有商店后14天的值
    :param all_data:
    :param save_filename: trainAsTest为true,则存储predict+real的值,前面14列为predict,后面14列为real,否则只有前面14列
    :param trainAsTest: 是否把训练集后14天当作测试集
    :param region: shopid区域，list,[startid,endid]
    :param predict_function: 预测单一商店的函数
    :param epoch_n: 每个商店循环多少次,从中取出最好的模型进行预测
    :return:
    """
    if predict_function is None:
        raise Exception("predict_function is none")
        return

    if region is None:
        startid = 1
        endid = 2000
    else:
        startid = region[0]
        endid = region[1]
    size = endid - startid + 1

    if not trainAsTest:
        result = np.zeros((size, 14))
        train_result = np.zeros((size, 28))
    else:
        result = np.zeros((size, 28))
    real = np.ndarray(0)
    for i in range(startid, endid + 1, 1):
        shopid = i
        print "shopid:", shopid
        best_score = 1
        best_model = None
        for j in range(epoch_n):
            train_predict, train_real_14, model = predict_function(shopid, all_data, True, None)
            scoreoneshop1 = scoreoneshop(train_predict, train_real_14)
            if(scoreoneshop1<best_score):
                best_score = scoreoneshop1
                best_model = model
        print "best:", best_score
        predict, real_14, model = predict_function(shopid, all_data, trainAsTest, best_model)
        # predict = real_14 = np.arange(14)
        if trainAsTest:
            real = np.append(real, real_14)
            predict = np.append(predict,real_14)
        train_predict_real = np.append(train_predict, train_real_14)
        result[i-startid] = predict
        train_result[i-startid] = train_predict_real
        gc.collect()


    result = pd.DataFrame(result.astype(np.int))
    train_result = pd.DataFrame(train_result.astype(int))
    result.insert(0, "id", value=range(startid, endid + 1, 1))
    train_result.insert(0, "id", value=range(startid, endid + 1, 1))
    # print result
    result = result.values
    train_result = train_result.values
    if(save_filename is not None):
        np.savetxt(save_filename, result, delimiter=",", fmt='%d')
        np.savetxt(save_filename+".train", train_result, delimiter=",", fmt='%d')
    else:
        print result
    return result


def predict_all_LSTM(all_data, save_filename, trainAsTest=False, region=None):
    """
    线性模型预测所有商店后14天的值
    :param all_data:
    :param save_filename:
    :param trainAsTest: 是否把训练集后14天当作测试集
    :param region: shopid区域，list,[startid,endid]
    :return:
    """
    predict_all(all_data,save_filename,trainAsTest,region,predictOneShop_LSTM)


def thread_worker(startid, num, all_data, q,lock):
    result = np.zeros((num, 15))
    for i in range(num):
        shopid = startid + i
        lock.acquire()
        print "shopid:", shopid
        lock.release()
        predict = predictOneShop_LSTM(shopid, all_data, False)[0]
        # predict = np.arange(14)
        predict = np.insert(predict, 0, shopid)
        result[i] = predict
    q.put(result)

def predict_all_LSTM_multithreads(all_data, save_filename, threadNum):
    """
    线性模型预测所有商店后14天的值
    :param all_data:
    :param save_filename:
    :param trainAsTest: 是否把训练集后14天当作测试集
    :param threadNum: 线程数量
    :return:
    """
    q = Queue.Queue()
    threads = []
    lock = threading.RLock()
    for i in range(threadNum):
        num = 2000/threadNum
        threads.append(threading.Thread(target=thread_worker, args=((i * num + 1), num, all_data, q, lock)))
    for i in range(threadNum):
        threads[i].setDaemon(True)
        threads[i].start()
    #等待所有线程完成
    for i in range(threadNum):
        threads[i].join()

    print "all finish"
    results = []
    while not q.empty():
        results.append(pd.DataFrame(q.get().astype(int)))
    result = pd.concat(results).sort_values(by=0).values
    if(save_filename is not None):
        np.savetxt(save_filename, result, delimiter=",", fmt='%d')
    else:
        print result
    return result

if __name__ == "__main__":
    from sys import argv
    # startid = int(argv[1])
    # endid = int(argv[2])
    pay_info = pd.read_csv(Parameter.payAfterGroupingAndRevisionAndCompletion_path)
    print predictOneShop_LSTM(6, pay_info, True)
    # predict_all_LSTM(pay_info, Parameter.projectPath + 'result/lstm_14f_train_%d.csv' % startid, True, [startid, endid])
    # predict_all_LSTM_multithreads(pay_info, Parameter.projectPath + "result/lstm_14f", 20)


