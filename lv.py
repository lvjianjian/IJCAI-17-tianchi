#-*- coding=gbk -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  Lasso
from sklearn.linear_model import LassoCV
from FeatureExtractor import *
from cjx_predict import scoreoneshop
import datetime

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

def predictOneShop_Lasso(shop_feature_path, feature_size):
    """
    L1正则化线性模型预测单个商店后14天的值
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
    #构造4个特征，分别是上周那天的值，上上周那天的值，上周平均值和上周标准差
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
    # clf.fit(train_x, train_y)
    #标准化处理器
    scaler = StandardScaler().fit(train_x)
    #寻找最优正则化参数值
    lassocv = LassoCV()
    lassocv.fit(train_x, train_y)
    #L1正则化线性模型
    lasso = Lasso(alpha=lassocv.alpha_)
    train_x = scaler.transform(train_x)
    lasso.fit(train_x, train_y)
    # print test_x1
    #先预测周二至周六的值
    test_x1 = scaler.transform(test_x1)
    test_y1 = lasso.predict(test_x1)
    # print test_y1
    #加上周一的值，计算均值和标准差
    week_count = np.insert(test_y1,0,feature["count"][n_sample - 1])
    #对预测的一周额外进行滤波修正
    """这里多加了预测值额外滤波修正"""
    for i in range(len(week_count)):
        week_count[i] = (week_count[i]+pays.ix[n_sample-2][1:8][i])/2
    #对预测的一周额外进行滤波修正结束
    week_mean = week_count.mean()
    week_std = week_count.std()
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
            test_x2[i][3] = week_std
    test_x2 = scaler.transform(test_x2)
    test_y2 = lasso.predict(test_x2)
    # test_y2 = clf.predict(test_x2)
    week_count2 = test_y2.copy()
    #对预测的一周额外进行滤波修正
    """这里多加了预测值额外滤波修正"""
    last_weekday = np.insert(test_y1,0,feature["count"][n_sample - 1])
    for i in range(len(week_count2)):
        week_count2[i] = (week_count2[i]+last_weekday[i])/2
    #对预测的一周额外进行滤波修正结束
    week_mean = week_count2.mean()
    week_std = week_count2.std()
    #最后预测最后一个周一的值
    test_x3 = np.zeros((1,feature_size))
    if feature_size >= 2:
        test_x3[0][0] = mean if isInvalid(week_count2[0]) else week_count2[0]
        test_x3[0][1] = mean if isInvalid(test_x2[0][0]) else test_x2[0][0]
    if feature_size == 4:
        test_x3[0][2] = week_mean
        test_x3[0][3] = week_std
    # test_y3 = clf.predict(test_x3)
    test_x3 = scaler.transform(test_x3)
    test_y3 = lasso.predict(test_x3)
    # last_y = np.insert(test_y1,len(test_y1),test_y2)
    # last_y = np.insert(last_y,len(last_y),test_y3)
    """这里用预测修正后的值作为最后结果"""
    week_count = week_count[1:7]
    last_y = np.insert(week_count,len(week_count),week_count2)
    last_y = np.insert(last_y,len(last_y),(test_y3 + test_y2[0])/2)
    return removeNegetive(toInt(last_y))


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
    std=feature["count"].std()
    #构造4个特征，分别是上周那天的值，上上周那天的值，上周平均值和上周标准差
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
    # print test_x1
    #先预测周二至周六的值
    test_y1 = clf.predict(test_x1)
    # print test_y1
    #加上周一的值，计算均值和标准差
    week_count = np.insert(test_y1,0,feature["count"][n_sample - 1])
    #对预测的一周额外进行滤波修正
    """这里多加了预测值额外滤波修正"""
    for i in range(len(week_count)):
        week_count[i] = (week_count[i]+pays.ix[n_sample-2][1:8][i])/2
    #对预测的一周额外进行滤波修正结束
    week_mean = week_count.mean()
    week_std = week_count.std()
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
            test_x2[i][3] = week_std
    test_y2 = clf.predict(test_x2)
    week_count2 = test_y2.copy()
    #对预测的一周额外进行滤波修正
    """这里多加了预测值额外滤波修正"""
    last_weekday = np.insert(test_y1,0,feature["count"][n_sample - 1])
    for i in range(len(week_count2)):
        week_count2[i] = (week_count2[i]+last_weekday[i])/2
    #对预测的一周额外进行滤波修正结束
    week_mean = week_count2.mean()
    week_std = week_count2.std()
    #最后预测最后一个周一的值
    test_x3 = np.zeros((1,feature_size))
    if feature_size >= 2:
        test_x3[0][0] = mean if isInvalid(week_count2[0]) else week_count2[0]
        test_x3[0][1] = mean if isInvalid(test_x2[0][0]) else test_x2[0][0]
    if feature_size == 4:
        test_x3[0][2] = week_mean
        test_x3[0][3] = week_std
    test_y3 = clf.predict(test_x3)
    # last_y = np.insert(test_y1,len(test_y1),test_y2)
    # last_y = np.insert(last_y,len(last_y),test_y3)
    """这里用预测修正后的值作为最后结果"""
    week_count = week_count[1:7]
    last_y = np.insert(week_count,len(week_count),week_count2)
    last_y = np.insert(last_y,len(last_y),(test_y3 + test_y2[0])/2)
    return removeNegetive(toInt(last_y))

def predict_all_Lasso(version,feature_size,save_filename):
    """
    线性模型预测所有商店后14天的值
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
            predict = predictOneShop_Lasso(path + "/" + filename, feature_size)
            result[i] = np.insert(predict,0,id)
            i += 1
    result = pd.DataFrame(result.astype(np.int))
    result = result.sort_values(by=0).values
    if(save_filename is not None):
        np.savetxt(save_filename,result,delimiter=",",fmt='%d')
    else:
        print result
    return result


def predict_all(version,feature_size,save_filename):
    """
    线性模型预测所有商店后14天的值
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
    if(save_filename is not None):
        np.savetxt(save_filename,result,delimiter=",",fmt='%d')
    else:
        print result
    return result


def predictOneShopInTrain(shop_feature_path, feature_size):
    """
    线性模型预测单个商店
    用训练集非后14天为训练集，预测后14天的值
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
    clf.fit(train_x, train_y)
    return [toInt(clf.predict(test_x)),test_y]


def predict_all_in_train(version, feature_size,save_filename = None):
    """
    线性模型预测所有商店
    用训练集非后14天为训练集，预测后14天的值
    :param version:
    :param feature_size:
    :return:
    """
    food_path="food_csvfile" + str(version)
    other_path = "other_csvfile" + str(version)
    market_path = "supermarket_csvfile" + str(version)
    paths = [food_path,other_path,market_path]
    result = np.zeros((2000,15))
    i = 0
    import os
    real = None
    predict = None
    for path in paths:
        csvfiles = os.listdir(path)
        for filename in csvfiles:
            id = int(filename.split("_")[0])
            predictAndReal = predictOneShopInTrain(path + "/" + filename, feature_size)
            if real is None:
                real = predictAndReal[1].values
            else:
                real = np.insert(real,len(real),predictAndReal[1].values)
            if predict is None:
                predict = predictAndReal[0]
            else:
                predict = np.insert(predict,len(predict),predictAndReal[0])
            result[i] = np.insert(predictAndReal[0], 0, id)
            i += 1
    result = pd.DataFrame(result.astype(np.int))
    result = result.sort_values(by=0).values
    if(save_filename is not None):
        np.savetxt(save_filename, result, delimiter=",", fmt='%d')
    return [predict, real, result]



def predict_all_LR_Split(all_data, save_filename,trainAsTest = False):
    """
    线性模型预测所有商店后14天的值
    :param all_data:
    :param save_filename:
    :param trainAsTest: 是否把训练集后14天当作测试集
    :return:
    """
    result = np.zeros((2000, 14))
    real = np.ndarray(0)
    for i in range(2000):
        shopid = i + 1
        print "shopid:",shopid
        predict, real_14 = predictOneTrain_LR_Split(shopid, all_data, trainAsTest)
        if trainAsTest:
            real = np.append(real, real_14)
        result[i] = predict
    if trainAsTest:
        predict = result.reshape((200*14))
        print "the final score : ", score(predict, real)
    result = pd.DataFrame(result.astype(np.int))
    result.insert(0, "id",value=range(1,2001,1))
    result = result.values
    if(save_filename is not None):
        np.savetxt(save_filename, result, delimiter=",", fmt='%d')
    else:
        print result
    return result


def predictOneTrain_LR_Split(shopid, all_data, trainAsTest=False):
    """
    一个商店分7天模型
    :param shopid:
    :param all_data:
    :param trainAsTest: 是否把训练集后14天当作测试集
    :return:如果trainAsTset是True,则返回[predicts,reals]，否则返回[predicts,None]
    """
    part_data = all_data[all_data.shopid == shopid]
    last_14_real_y = None
    # 取出一部分做训练集
    if trainAsTest: #使用训练集后14天作为测试集的话，训练集为前面部分
        part_data = part_data[0:len(part_data) - 14]
        last_14_real_y = part_data[len(part_data) - 14:]["count"].values

    skipNum = 0
    sameday = extractBackSameday(part_data, 2, skipNum, nan_method_sameday_mean)
    count = extractCount(part_data, skipNum)
    model = []
    part_counts = []
    #lr model fit
    for i in range(7):
        lr = LinearRegression()
        model.append(lr)
        weekday = i + 1
        part_sameday = getOneWeekdayFomExtractedData(sameday, weekday)
        part_count = getOneWeekdayFomExtractedData(count, weekday)
        part_counts.append(part_count)
        lr.fit(part_sameday, part_count)

    #lr model predict
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
        index = getWeekday(strftime) - 1
        part_count = part_counts[index]
        #取前2周同一天的值为特征进行预测
        x = [[part_count[len(part_count) - 1][0], part_count[len(part_count) - 2][0]]]
        predict = model[index].predict(x)
        preficts.append(predict[0][0])
        part_counts[index] = np.append(part_count, predict).reshape((part_count.shape[0] + 1, 1))
    preficts = (removeNegetive(toInt(np.array(preficts)))).astype(int)
    if trainAsTest:
        last_14_real_y = (removeNegetive(toInt(np.array(last_14_real_y)))).astype(int)
        # print preficts,last_14_real_y
        print str(shopid)+',score:', scoreoneshop(preficts, last_14_real_y)
    return [preficts, last_14_real_y]


if __name__ == "__main__":
    import Parameter
    #读取的是补全的商店信息
    data = pd.read_csv(Parameter.payAfterGroupingAndRevision2AndTurncate_path)
    # predict_all_LR_Split(data, "result/lr_split_2f.csv", False)
    # print predictOneTrain_LR_Split(1, data, True)
    predict_all(version=3, feature_size=2, save_filename="result/result_meanfilter_extra_resultfilter_f2.csv")
    # print predictOneShop("food_csvfile3/1_trainset.csv",feature_size=2)
    # print predictOneShopInTrain("food_csvfile2/1243_trainset.csv",feature_size=2)[0]
    # prediceAndReal = predict_all_in_train(version=3, feature_size=4)
    # print score(prediceAndReal[0], prediceAndReal[1])
    # print prediceAndReal[2]