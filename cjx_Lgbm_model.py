# encoding=utf-8

# Naive LSTM to learn three-char time steps to one-char mapping
import numpy as np
import Parameter as para
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error

# encoding=utf-8

# Naive LSTM to learn three-char time steps to one-char mapping
cur_thread_num = 20;
# 序列长度设为7
seq_length = 14
dateparser1 = para.dateparse1
meanfiltered_data = pd.read_csv ('processing_files/meanfiltered.csv', date_parser=dateparser1)


def getCoutList (traindata, shopId):
    '''

    :param traindata:单个店家的训练数据
    :param shopId: 商店ID
    :return: 商店ID对应的count序列（最好保持为float类型）
    '''
    countList = map (float, traindata[traindata['shopid'] == shopId]['count'].values)
    return countList


def preprocessCoutList (seq_length, counList):
    '''

    :param seq_length: 时间轴辐射长度
    :param counList:
    :return:
    '''
    dataX = []
    dataY = []
    # 将一天前两周时间的值化为特征向量组
    # 当天的值作为样本结果
    for i in range (0, len (counList) - seq_length, 1):
        seq_in = counList[i:i + seq_length]
        seq_out = counList[i + seq_length]
        dataX.append (seq_in)
        dataY.append (seq_out)
        # print seq_in, '->', seq_out
    return [dataX, dataY]


def predictInTrainOneShop_LSTM (train_data, seq_length, id_item):
    countList = getCoutList (train_data, id_item)
    [train_feature, train_label] = preprocessCoutList (seq_length, countList)
    # 生成测试集合
    [test_feature, test_label] = preprocessCoutList (seq_length,
                                                     countList[len (countList) - 2 * seq_length:len (countList)])
    # create dataset for lightgbm
    lgb_train = lgb.Dataset (train_feature, train_label)
    # create dataset for lightgbm
    lgb_eval = lgb.Dataset (test_feature, test_label, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2'},   
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    print('Start training...')
    # train
    gbm = lgb.train (params,
                     lgb_train,
                     num_boost_round=50,
                     valid_sets=lgb_eval)
                     # early_stopping_rounds=5)
    print('Save model...')
    # save model to file
    gbm.save_model ('model/' + str (id_item) + 'lightgbm_model.txt')

    print('Start predicting...')
    # predict
    y_pred = gbm.predict (test_feature, num_iteration=gbm.best_iteration)
    # 返回两个一行的向量(1,14)
    print '##############'+str(id_item)+'############'
    print 'pred:',y_pred
    print 'real:',test_label
    print 'score:',scoreoneshop(y_pred,test_label)
    return [y_pred, test_label]


def predict_All_inTrain (train_data, seq_length, save_filename):
    result = np.zeros ((2000, 15))
    i = 0
    import os
    real = None
    predict = None
    shopid_values = train_data['shopid'].unique()
    print shopid_values
    for i, sid in enumerate (shopid_values):
        print sid
        # sid=int(sid)
        predictAndReal = predictInTrainOneShop_LSTM (train_data, seq_length, sid)
        if real is None:
            real = predictAndReal[1]
        else:
            real = np.insert (real, len (real), predictAndReal[1])
        if predict is None:
            predict = predictAndReal[0]
        else:
            predict = np.insert (predict, len (predict), predictAndReal[0])
    result[i] = np.insert (predictAndReal[0], 0, sid)
    result = pd.DataFrame (result.astype (np.int))
    result = result.sort_values (by=0).values
    if (save_filename is not None):
        np.savetxt (save_filename, result, delimiter=",", fmt='%d')

    return [predict, real, result]


def scoreoneshop(predict, real):
    """
    评测公式
    :param predict: 预测值
    :param real: 真实值
    :return: 得分
    """
    # print "predict:", predict
    # print "real:", real
    score = 0
    predict=np.array(predict)
    real=np.array(real)
    for i in range (14):
        score += (abs (predict[i] - real[i]) / (predict[i] + real[i]))
    score /= 14
    return score


def score (predict, real):
    """
    评测公式
    :param predict: 预测值
    :param real: 真实值
    :return: 得分
    """
    # print "predict:", predict
    # print "real:", real
    score = 0
    for i in range (predict.shape[0]):
        score += (abs (predict[i] - real[i]) / (predict[i] + real[i]))
    score /= predict.shape[0]
    return score


if __name__ == '__main__':
    '''
    '''
    prediceAndReal = predict_All_inTrain (meanfiltered_data, seq_length, 'result\\result_train_lgbm.csv')
    print score (prediceAndReal[0], prediceAndReal[1])
