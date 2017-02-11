#-*- coding=gbk -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  Lasso
from sklearn.linear_model import LassoCV

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
    # print "predict:", predict
    # print "real:", real
    score = 0
    for i in range(predict.shape[0]):
        score += (abs(predict[i]-real[i])/(predict[i]+real[i]))
    score /= predict.shape[0]
    return score


def toInt(x):
    """
    将ndarray中的数字四舍五入
    :param x:
    :return:
    """
    for i in range(x.shape[0]):
        x[i] = int(round(x[i]))
    return x


def predictOneShopInTrain_Lasso(shop_feature_path, feature_size):
    """
    线性模型预测单个商店
    用训练集非后14天为训练集，预测后14天的值
    :param shop_feature_path:
    :param feature_size:
    :return:
    """
    #线性回归
    # clf = LinearRegression()
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
    std=feature["count"].std()
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
            last_std = pays.ix[i][1:8].std()#计算上一周方差
            x[i][3] = std if isInvalid(last_std) else last_std

    train_x = x[:x.shape[0]-14]
    test_x = x[x.shape[0]-14:]
    train_y = feature["count"][:x.shape[0]-14]
    test_y = feature["count"][x.shape[0]-14:]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    lassocv = LassoCV()
    lassocv.fit(train_x, train_y)
    lasso = Lasso(alpha=lassocv.alpha_)
    lasso.fit(train_x, train_y)

    # print "Lasso model: ", lasso.coef_
    # clf.fit(train_x, train_y)
    return [lasso.predict(test_x),test_y]


def predict_all_in_train_Lasso(version, feature_size):
    """
    线性模型预测所有商店
    用训练集非后14天为训练集，预测后14天的值
    :param version:
    :param feature_size:
    :return:
    """
    food_path="food_csvfile_holiday" + str(version)
    other_path = "other_csvfile_holiday" + str(version)
    market_path = "supermarket_csvfile_holiday" + str(version)
    paths = [food_path,other_path,market_path]
    import os
    real = None
    predict = None
    for path in paths:
        csvfiles = os.listdir(path)
        for filename in csvfiles:
            predictAndReal = predictOneShopInTrain_Lasso(path + "/" + filename, feature_size)
            if real is None:
                real = predictAndReal[1].values
            else:
                real = np.insert(real,len(real),predictAndReal[1].values)
            if predict is None:
                predict = predictAndReal[0]
            else:
                predict = np.insert(predict,len(predict),predictAndReal[0])
    return [predict, real]



if __name__ == "__main__":
    # prediceAndReal=predict_all(version=2,feature_size=4,save_filename="result/result_revise_f4.csv")
    # print predictOneShop("food_csvfile2/1243_trainset.csv",feature_size=4)
    prediceAndReal = predict_all_in_train_Lasso(version=2, feature_size=4)
    print score(prediceAndReal[0], prediceAndReal[1])