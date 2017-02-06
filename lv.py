#-*- coding=gbk -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



def isInvalid(value):
    """
    check value whether inValid or not
    :param value:
    :return: nan or 0 is invalid and return true ,otherwise
    """
    if np.isnan(value) or value == 0:
        return True
    else:
        return False

def score(predict,real):
    """
    评测公式
    :param predict: 预测值
    :param real: 真实值
    :return: 得分
    """
    print "predict:", predict
    print "real:", real
    score = 0
    for i in range(predict.shape[0]):
        score += (abs(predict[i]-real[i])/(predict[i]+real[i]))
    score /= predict.shape[0]
    return score

def predictOneShop(shop_feature_path):
    #线性回归
    clf = LinearRegression()
    clf.normalize=True
    feature = pd.read_csv(shop_feature_path)

    #上周一到周7各列各列，对应下标1到7
    pays = feature[["count","pay_day1","pay_day2","pay_day3","pay_day4","pay_day5","pay_day6","pay_day7"]]
    #周几那列
    weekday=feature["weekday"]
    #上上周那列
    same_day=feature["same_day"]


    n_sample = feature.shape[0]
    x = np.zeros((n_sample, 4))

    mean=feature["count"].mean()
    var=feature["count"].var()
    #构造4个特征，分别是上周那天的值，上上周那天的值，上周平均值和上周方差
    for i in range(n_sample):
        day = weekday[i]
        last_pay = pays.ix[i][day] #上周那天的值
        x[i][0] = mean if isInvalid(last_pay)  else last_pay #无效暂由均值替代
        x[i][1] = mean if isInvalid(same_day[i]) else same_day[i] #上上周那天的值
        last_mean = pays.ix[i][1:8].mean() #计算上一周平均值
        x[i][2] = mean if isInvalid(last_mean) else last_mean
        last_var = pays.ix[i][1:8].var()#计算上一周方差
        x[i][3] = var if isInvalid(last_var) else last_var

    train_x = x[:x.shape[0]-7]
    test_x = x[x.shape[0]-7:]
    train_y = feature["count"][:x.shape[0]-7]
    test_y = feature["count"][x.shape[0]-7:]
    # train_x = np.concatenate([train_x, train_x], axis = 0)
    # train_y = np.concatenate([train_y, train_y], axis = 0)
    clf.fit(train_x, train_y)
    return [clf.predict(test_x),test_y]

def predictOneShop(shop_feature_path,feature_size):
    #线性回归
    clf = LinearRegression()
    feature = pd.read_csv(shop_feature_path)

    #上周一到周7各列各列，对应下标1到7
    pays = feature[["count","pay_day1","pay_day2","pay_day3","pay_day4","pay_day5","pay_day6","pay_day7"]]
    #周几那列
    weekday=feature["weekday"]
    #上上周那列
    same_day=feature["same_day"]


    n_sample = feature.shape[0]
    x = np.zeros((n_sample, feature_size))

    mean=feature["count"].mean()
    var=feature["count"].var()
    #构造4个特征，分别是上周那天的值，上上周那天的值，上周平均值和上周方差
    for i in range(n_sample):
        day = weekday[i]
        last_pay = pays.ix[i][day] #上周那天的值
        if feature_size >= 2:
            x[i][0] = mean if isInvalid(last_pay)  else last_pay #无效暂由均值替代
            x[i][1] = mean if isInvalid(same_day[i]) else same_day[i] #上上周那天的值
        if feature_size == 4:
            last_mean = pays.ix[i][1:8].mean() #计算上一周平均值
            x[i][2] = mean if isInvalid(last_mean) else last_mean
            last_var = pays.ix[i][1:8].var()#计算上一周方差
            x[i][3] = var if isInvalid(last_var) else last_var

    train_x = x[:x.shape[0]-7]
    test_x = x[x.shape[0]-7:]
    train_y = feature["count"][:x.shape[0]-7]
    test_y = feature["count"][x.shape[0]-7:]
    # train_x = np.concatenate([train_x, train_x], axis = 0)
    # train_y = np.concatenate([train_y, train_y], axis = 0)
    clf.fit(train_x, train_y)
    return [clf.predict(test_x),test_y]

if __name__ == "__main__":
    # predictAndReal = predictOneShop("supermarket1_csvfile/2trainset.csv",4)
    # print score(predictAndReal[0], predictAndReal[1].values)
    pay_info = pd.read_csv("data/user_pay_afterGrouping.csv")
    max = 0
    for i in range(1,2000,1):
        v = pay_info[pay_info.shopid == i].shape[0]
        print v
        if v > max:
            max = v
    print max