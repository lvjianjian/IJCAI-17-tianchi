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

def predictOneShop(shop_feature_path, feature_size):
    """
    线性模型预测单个商店后14天的值
    :param shop_feature_path:
    :param feature_size:
    :return:
    """
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

    train_x = x[:]
    train_y = feature["count"][:]

    #提取要预测的14天的特征，先提取前6天的值,也就是星期2到星期7
    test_x1 = np.zeros((6, feature_size))

    #预测第一天为周二，使用周一的数据即可
    for i in range(6):
        last_pay = pays.ix[n_sample - 1]["pay_day"+str(i+2)]
        if feature_size >= 2:
            test_x1[i][0] = mean if isInvalid(last_pay) else last_pay #无效暂由均值替代
            test_x1[i][1] = mean if isInvalid(x[n_sample - 7 + i][0]) else x[n_sample - 7 + i][0] #上上周那天的值
        if feature_size == 4:
            test_x1[i][2] = x[n_sample - 1][2]
            test_x1[i][3] = x[n_sample - 1][3]
    clf.fit(train_x, train_y)
    #先预测周二至周六的值
    test_y1 = clf.predict(test_x1)
    #加上周一的值，计算均值和方差
    week_count = np.insert(test_y1,0,feature["count"][n_sample - 1])
    week_mean = week_count.mean()
    week_var = week_count.var()
    #接下来的周一到周7
    test_x2 = np.zeros((7,feature_size))
    for i in range(7):
        last_pay = week_count[i]
        if i == 0:
            last_last_pay = x[n_sample - 1][0]
        else:
            last_last_pay = test_x1[i - 1][0]
        if feature_size >= 2:
            test_x2[i][0] = mean if isInvalid(last_pay) else last_pay #无效暂由均值替代
            test_x2[i][1] = mean if isInvalid(last_last_pay) else last_last_pay #上上周那天的值
        if feature_size == 4:
            test_x2[i][2] = week_mean
            test_x2[i][3] = week_var
    text_y2 = clf.predict(test_x2)
    week_mean = text_y2.mean()
    week_var = text_y2.var()
    #最后预测最后一个周一的值
    test_x3 = np.zeros((1,feature_size))
    if feature_size >= 2:
        test_x3[0][0] = mean if isInvalid(text_y2[0]) else text_y2[0]
        test_x3[0][1] = mean if isInvalid(test_x2[0][0]) else test_x2[0][0]
    if feature_size == 4:
        test_x3[0][2] = week_mean
        test_x3[0][3] = week_var
    test_y3 = clf.predict(test_x3)
    last_y = np.insert(test_y1,len(test_y1),text_y2)
    last_y = np.insert(last_y,len(last_y),test_y3)
    return toInt(last_y)


def predict_all(version,feature_size,save_filename):
    """
    线性模型预测单个商店后14天的值
    :param version:
    :param feature_size:
    :param save_filename:
    :return:
    """
    food_path="food_csvfile" + str(version)
    other_path = "other_csvfile" + str(version)
    market_path = "supermarket_csvfile" + str(version)
    paths = [food_path,other_path,market_path]
    result = np.zeros((2000,15))
    i = 0
    import os
    for path in paths:
        csvfiles = os.listdir(path)
        for filename in csvfiles:
            id = int(filename.split("_")[0])
            predict = predictOneShop(path + "/" + filename, feature_size)
            result[i] = np.insert(predict,0,id)
            i += 1
    result = pd.DataFrame(result.astype(np.int))
    result = result.sort_values(by=0).values
    np.savetxt(save_filename,result,delimiter=",",fmt='%d')


def predictOneShopInTrain(shop_feature_path, feature_size):
    """
    线性模型预测单个商店
    用训练集非后7天为训练集，预测后7天的值
    :param shop_feature_path:
    :param feature_size:
    :return:
    """
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
    clf.fit(train_x, train_y)
    return [clf.predict(test_x),test_y]


def predict_all_in_train(version, feature_size):
    """
    线性模型预测所有商店
    用训练集非后7天为训练集，预测后7天的值
    :param version:
    :param feature_size:
    :return:
    """
    food_path="food_csvfile" + str(version)
    other_path = "other_csvfile" + str(version)
    market_path = "supermarket_csvfile" + str(version)
    paths = [food_path,other_path,market_path]
    import os
    real = None
    predict = None
    for path in paths:
        csvfiles = os.listdir(path)
        for filename in csvfiles:
            predictAndReal = predictOneShopInTrain(path + "/" + filename, feature_size)
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
    predict_all(version=2,feature_size=2,save_filename="result/result_revise_f2.csv")
    # prediceAndReal = predict_all_in_train(version=2, feature_size=4)
    # print score(prediceAndReal[0], prediceAndReal[1])