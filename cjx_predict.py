# encoding=utf-8

# Naive LSTM to learn three-char time steps to one-char mapping
from keras.optimizers import SGD
import numpy as np
from keras.optimizers import RMSprop
from keras.regularizers import l2, activity_l2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
import pandas as pd
import Parameter as para
import  threading
cur_thread_num=20;
# 序列长度设为7
seq_length = 14
dateparser1=para.dateparser1

def toInt(x):
    """
    将ndarray中的数字四舍五入
    :param x:
    :return:
    """
    x=int(round(x))
    return x

def getCoutList(traindata,shopId):
    '''

    :param traindata:单个店家的训练数据
    :param shopId: 商店ID
    :return: 商店ID对应的count序列（最好保持为float类型）
    '''
    countList=map(float,traindata[traindata['shopid']==shopId]['count'].values)
    return countList

def preprocessCoutList(seq_length,counList):
    '''

    :param seq_length: 时间轴辐射长度
    :param counList:
    :return:
    '''
    dataX=[]
    dataY=[]
    # 将一天前两周时间的值化为特征向量组
    # 当天的值作为样本结果
    for i in range(0, len(counList) - seq_length, 1):
        seq_in = counList[i:i + seq_length]
        seq_out = counList[i + seq_length]
        dataX.append(seq_in)
        dataY.append(seq_out)
        # print seq_in, '->', seq_out
    dataX=np.reshape(dataX, (len(dataX), seq_length, 1))
    return [dataX,dataY]

def predictInTrainOneShop_lgbm(train_data,seq_length,id_item):
    shopid_values=train_data['shopid'].values
    all_countList = getCoutList(train_data, id_item)
    # 取出一部分做训练集
    part_countList = all_countList[0:len(all_countList)]
    [train_x, train_y] = preprocessCoutList(seq_length, part_countList)
    test_coutList = all_countList[len(all_countList) - 2*seq_length:len(all_countList)]
    [test_x, test_y] = preprocessCoutList(seq_length, test_coutList)

    model = Sequential()
    model.add(LSTM(32, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1, activation='linear',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)
                    )
              )

    # 设置优化器（除了学习率外建议保持其他参数不变）
    rms=RMSprop(lr=0.1, rho=0.9, epsilon=1e-06)
    # sgd=SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss='mean_absolute_percentage_error', optimizer=rms, metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x , train_y, nb_epoch=2, batch_size=1, verbose=2)
    prediction_v=[]
    prediction = model.predict(test_x, verbose=2)
    for itm in enumerate(prediction):
        prediction_v.append(toInt(float(itm[0])))
    # print np.array(prediction_v),np.array(test_y)
    print str(id_item)+'score:',scoreoneshop(np.array(prediction_v), np.array(test_y))

    return [prediction_v,test_y,id_item]

def predict_All_inTrain(train_data,seq_length,save_filename):

    result = np.zeros((2000, 15))
    i = 0
    import os
    real = None
    predict = None
    shopid_values=train_data['shopid'].values

    for sid in shopid_values:
        predictAndReal = predictInTrainOneShop_lgbm(train_data, seq_length,sid)
        if real is None:
            real = predictAndReal[1]
        else:
            real = np.insert(real, len(real), predictAndReal[1])
        if predict is None:
            predict = predictAndReal[0]
        else:
            predict = np.insert(predict, len(predict), predictAndReal[0])
        result[sid] = np.insert(predictAndReal[0], 0, id)
    result = pd.DataFrame(result.astype(np.int))
    result = result.sort_values(by=0).values
    if (save_filename is not None):
        np.savetxt(save_filename, result, delimiter=",", fmt='%d')

    return [predict, real, result]


def scoreoneshop(predict,real):
    """
    评测公式
    :param predict: 预测值
    :param real: 真实值
    :return: 得分
    """
    # print "predict:", predict
    # print "real:", real
    score = 0
    for i in range(14):
        score += (abs((float)(predict[i]-real[i]))/(predict[i]+real[i]))
    score /= 14
    return score

def score(predict,real):
    """
    评测公式
    :param predict: 预测值
    :param real: 真实值
    :return: 得分
    """
    # print "predict:", predict
    # print "real:", real
    score = 0
    for i in range(predict.shape[0]):
        score += (abs((float)(predict[i]-real[i]))/(predict[i]+real[i]))
    score /= predict.shape[0]
    return score


if __name__=='__main__':
    '''
    '''
    meanfiltered_data=pd.read_csv('processing_files/meanfiltered.csv')
    prediceAndReal=predict_All_inTrain(meanfiltered_data,seq_length,'result\\result_train_lstm.csv')
    print score(prediceAndReal[0], prediceAndReal[1])