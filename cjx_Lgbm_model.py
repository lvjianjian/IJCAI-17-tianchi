# encoding=utf-8

# Naive LSTM to learn three-char time steps to one-char mapping
import numpy as np
import Parameter as para
import json
import lightgbm as lgb
import pandas as pd
# from lv import  toInt
# from  lv import  removeNegetive

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# encoding=utf-8

# Naive LSTM to learn three-char time steps to one-char mapping
cur_thread_num = 20;
# 序列长度设为7
seq_length = 7
dateparser1 = para.dateparse1
#此文件已经做过均值平滑且填补完整
train_data = pd.read_csv ('data/user_pay_afterGroupingAndRevisionAndCompletion.csv', date_parser=dateparser1)


def toInt(x):
    """
    将ndarray中的数字四舍五入
    :param x:
    :return:
    """
    for i in range(x.shape[0]):
        x[i] = int(round(x[i]))
    return x


def removeNegetive(x):
    """
    去除负数，用1代替
    :param x:
    :return:
    """
    for i in range(x.shape[0]):
       if(x[i]<0):
           x[i] = 1
    return x

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




def predictInTrainOneShop_Lgbm (train_data, seq_length, id_item):
    countList = getCoutList (train_data, id_item)
    #这里的14是验证集合长度
    [train_feature, train_label] = preprocessCoutList (seq_length, countList[:len(countList)-14])
    feature_len=len(train_feature)
    print len(train_feature)
    train_len = feature_len * 2/3
    eval_len  = feature_len - train_len

    train_x=train_feature[:train_len]
    train_y=train_label[:train_len]
    eval_x = train_feature[train_len:]
    eval_y = train_label[train_len:]
    # 生成测试集合
    [test_feature, test_label] = preprocessCoutList (seq_length,
                                                     countList[len (countList) - 2 * 14:len (countList)])
    # [_feature, test_label] = preprocessCoutList (seq_length,
    #                                                  countList[len (countList) - 2 * 14:len (countList)])
    # create dataset for lightgbm
    # lgb_train = lgb.Dataset (train_feature, train_label)
    # create dataset for lightgbm
    # lgb_eval = lgb.Dataset (test_feature, test_label, reference=lgb_train)

    # specify your configurations as a dict
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'metric': {'l2','acc'},
    #
    #     'learning_rate' : 0.1,
    #     'num_leaves' :255,
    #     'num_trees ':500,
    #     'num_threads' :16,
    #     'min_data_in_leaf ':0,
    #     'min_sum_hessian_in_leaf' :100
    # }
    # print params
    print('Start training...')
    # train
    gbm = lgb.LGBMRegressor (objective='regression',
                             num_leaves=255,
                             # num_trees=500,
                             learning_rate=0.1,
                             n_estimators=30)

    gbm.fit (train_x, train_y,
             eval_set=[(eval_x, eval_y)],
             eval_metric='l2',
             early_stopping_rounds=15)
    # estimator = lgb.LGBMRegressor (num_leaves=31)
    #
    # param_grid = {
    #
    #     'learning_rate': [0.01, 0.1, 1],
    #
    #     'n_estimators': [20, 40]
    #
    # }
    #
    # gbm = GridSearchCV (estimator, param_grid)

    # gbm.fit (train_feature, train_label)
    # print('Best parameters found by grid search are:', gbm.best_params_)
    # print('Save model...')
    # save model to file
    # gbm.save_model ('model/' + str (id_item) + 'lightgbm_model.txt')

    print('Start predicting...')
    # predict
    y_pred = gbm.predict (test_feature, num_iteration=gbm.best_iteration)
    # 返回两个一行的向量(1,14)
    print '#########################'+str(id_item)+'#############################'
    # print 'features:',test_feature
    # print 'pred:',y_pred
    # print 'real:',test_label
    curscore=scoreoneshop (y_pred, test_label)
    print 'score:',curscore
    with open('LGBMProcessingfile/offline_resultlistseq7.txt','a') as rlistfile:
        rlistfile.write(str(id_item)+':'+str(curscore)+'\n')
    return [y_pred, test_label]


def predictInTrainOneShopOnline_Lgbm (train_data, seq_length, id_item):
    countList = getCoutList (train_data, id_item)
    [train_feature, train_label] = preprocessCoutList (seq_length, countList)
    feature_len = len (train_feature)
    print 'feature_len:',len (train_feature)
    train_len = feature_len * 2 / 3        #  训练集长度（这里可以更换训练集提取策略来提升模型性能）
    eval_len = feature_len - train_len     #  验证集合长度
    train_x = train_feature[:train_len]
    train_y = train_label[:train_len]
    eval_x = train_feature[train_len:]
    eval_y = train_label[train_len:]
    # 生成测试集合
    # [test_feature, test_label] = preprocessCoutList (seq_length,
    #                                                  countList[len (countList) - 2 * 14:len (countList)])
    # [_feature, test_label] = preprocessCoutList (seq_length,
    #                                              countList[len (countList) - 2 * 14:len (countList)])
    # create dataset for lightgbm
    # lgb_train = lgb.Dataset (train_feature, train_label)
    # create dataset for lightgbm
    # lgb_eval = lgb.Dataset (test_feature, test_label, reference=lgb_train)

    # specify your configurations as a dict
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'metric': {'l2','acc'},
    #
    #     'learning_rate' : 0.1,
    #     'num_leaves' :255,
    #     'num_trees ':500,
    #     'num_threads' :16,
    #     'min_data_in_leaf ':0,
    #     'min_sum_hessian_in_leaf' :100
    # }
    # print params
    print('Start training...')
    # train
    gbm = lgb.LGBMRegressor (objective='regression',
                             num_leaves=255,
                             # num_trees=500,
                             learning_rate=0.1,
                             n_estimators=30)

    gbm.fit (train_x, train_y,
             eval_set=[(eval_x, eval_y)],
             eval_metric='l2',
             early_stopping_rounds=10)
    # estimator = lgb.LGBMRegressor (num_leaves=31)
    #
    # param_grid = {
    #
    #     'learning_rate': [0.01, 0.1, 1],
    #
    #     'n_estimators': [20, 40]
    #
    # }
    #
    # gbm = GridSearchCV (estimator, param_grid)

    # gbm.fit (train_feature, train_label)
    # print('Best parameters found by grid search are:', gbm.best_params_)
    # print('Save model...')
    # save model to file
    # gbm.save_model ('model/' + str (id_item) + 'lightgbm_model.txt')

    print('Start predicting...')
    y_pre_list=[]
    test_feature_pool = countList[len (countList)-seq_length :
                                  len(countList)]          # 先取后14天的
    for  predict_day in range(14):
        y_pred = gbm.predict ([test_feature_pool], num_iteration=gbm.best_iteration)
        test_feature_pool=np.insert(test_feature_pool,len(test_feature_pool),y_pred[0])
        y_pre_list=np.insert(y_pre_list,len(y_pre_list),y_pred[0])
        test_feature_pool = test_feature_pool[1:len (test_feature_pool)]    # 截断成特征向量
    # 返回两个一行的向量(1,14)
    print '#########################' + str (id_item) + '#############################'
    y_pre_list=toInt(y_pre_list)
    y_pre_list=removeNegetive(y_pre_list)
    print 'y_pre_list:',y_pre_list
    return y_pre_list

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
        predictAndReal = predictInTrainOneShop_Lgbm (train_data, seq_length, sid)
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

def predict_All_online (train_data, seq_length, save_filename):
    i = 0
    shopid_values = train_data['shopid'].unique()
    result = np.zeros ((2000, 15))
    print shopid_values
    for i in range (0,2000):
        print 'i:',i
        # sid=int(sid)
        predict= predictInTrainOneShopOnline_Lgbm(train_data, seq_length, shopid_values[i])
        result[i] = np.insert (predict, 0, shopid_values[i])
        # print result[i]
    result = pd.DataFrame (result.astype (np.int))
    result = result.sort_values (by=0).values
    if (save_filename is not None):
        np.savetxt (save_filename, result, delimiter=",", fmt='%d')
    else:
        print result
    return result

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
    for i in range (7):
        score += (abs (predict[i] - real[i]) / (predict[i] + real[i]))
    score /= 7
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
    # print 'user_pay_afterGroupingAndRevisionAndCompletion.csv'
    # predictInTrainOneShopOnline_Lgbm(meanfiltered_data,7,1918)
    # prediceAndReal = predict_All_inTrain (train_data, seq_length, 'result/result_train_lgbm1_l2seq7v2.csv')
    # print score (prediceAndReal[0], prediceAndReal[1])
    print 'seq_length:',seq_length
    predict_All_online(train_data,seq_length,'result/predict_seq7_lgbm_1_3_255_nonmeanf.csv') #此为2017/2/6日晚上所生成的预测文件（loss 为 mse）
    #predictInTrainOneShop_Lgbm(meanfiltered_data, seq_length, 3)
#