# encoding=utf-8


import numpy as np
from DataRevision import getWeekday
import pandas as pd

nan_method_global_mean = "global_mean" #全局平均
nan_method_sameday_mean = "sameday_mean"#sameday平均


def getReplaceValue(replace,method,weekday):
    if method == nan_method_global_mean:
        return replace[0]
    elif method == nan_method_sameday_mean:
        return replace[weekday - 1]


def extractCount(part_data,skipNum = 0):
    """
    从{skipNum}开始抽取count值
    :param part_data:
    :param skipNum:
    :return: ndarray,shape(处理的样例出数,2),第一列为count，第二列为日期time
    """
    if(skipNum <0 or skipNum > part_data.shape[0]):
        raise Exception("parameter skipNum error")
        return
    dataY = []
    count = part_data["count"].values
    time = part_data["time"].values
    for i in range(skipNum,part_data.shape[0],1):
        dataY.append(count[i])
        dataY.append(time[i])
    dataY = np.reshape(dataY,(part_data.shape[0] - skipNum,2))
    return dataY


def extractBackDayByNCycle(part_data, backNum = 1, startNum = 0, nan_method=nan_method_global_mean,cycle = 1):
    """
    从{startNum}开始抽取前{backNum}的以{cycle}为周期的day的值, 排列按照前面第{cycle}天，前面第{2*cycle}天这样由近就及远排放
    :param part_data:需要含有两列组成，一列为count,另一列为time,注意这里的数据是要按日期补全的数据，不然会有不对应的情况
    :param backNum: 向前抽取多少天
    :param startNum: 前面跳过的样例数，也就是从第{startNum}开始生成
    :param nan_method: 缺失值处理方式
    :return: ndarray,shape(处理的样例出数,backNum + 1),第一列到第{backNum}列为前第{cycle}天的值到前第{backNum * cycle}天的值
            第{backNum}+1列为那个样例的日期
    """
    #check parameter
    if(not (nan_method == nan_method_global_mean or nan_method == nan_method_sameday_mean)):
        raise Exception("parameter nan_method error")
        return
    if(startNum <0 or startNum > part_data.shape[0]):
        raise Exception("parameter skipNum error")
        return
    if(backNum<1):
        raise Exception("parameter backNum error")
        return


    #先加入weekday一列，方便后面处理
    part_data.insert(3, "weekday", part_data["time"].map(getWeekday))


    #计算缺失值的代替值
    replace=[]
    if(nan_method == nan_method_global_mean):
        replace.append(part_data["count"].mean())
    elif(nan_method == nan_method_sameday_mean):
        for i in range(7):
            replace.append(part_data[part_data.weekday==(i+1)]["count"].mean())

    count = part_data["count"].values
    weekday = part_data["weekday"].values
    time = part_data["time"].values
    dataX=[]
    for i in range(startNum, part_data.shape[0], 1):
        for j in range(backNum):
            index = i - cycle * (j + 1)
            if index < 0:
                value = getReplaceValue(replace, nan_method, weekday[i])
            else:
                value = count[index]
            dataX.append(value)
        dataX.append(time[i])
    dataX = np.reshape(dataX, (part_data.shape[0] - startNum, backNum + 1))
    return dataX



def extractBackDay(part_data, backNum = 1, startNum = 0, nan_method=nan_method_global_mean):
    """
    从{startNum}开始抽取前{backNum}的day的值, 排列按照前面第一天，前面第二天这样由近就及远排放
    :param part_data:需要含有两列组成，一列为count,另一列为time,注意这里的数据是要按日期补全的数据，不然会有不对应的情况
    :param backNum: 向前抽取多少天
    :param startNum: 前面跳过的样例数，也就是从第{startNum}开始生成
    :param nan_method: 缺失值处理方式
    :return: ndarray,shape(处理的样例出数,backNum + 1),第一列到第{backNum}列为前第一天的值到前第{backNum}天的值
            第{backNum}+1列为哪个样例的日期
    """
    return extractBackDayByNCycle(part_data,backNum,startNum,nan_method,1)



def extractBackSameday(part_data, backNum = 1, startNum = 0, nan_method=nan_method_global_mean):
    """
    从{startNum}开始抽取前{backNum}周的sameday, 排列按照前一周的sameday，前第二周的sameday这样由近就及远排放
    :param part_data:需要含有两列组成，一列为count,另一列为time,注意这里的数据是要按日期补全的数据，不然会有不对应的情况
    :param backNum: 向前抽取多少周的sameday
    :param startNum: 前面跳过的样例数，也就是从第{startNum}开始生成
    :param nan_method: 缺失值处理
    :return: ndarray,shape(处理的样例出数,backNum + 1),第一列到第{backNum}列为前第一周的sameday的值到前第{backNum}周的sameday的值
            第{backNum}+1列为哪个样例的日期
    """
    return extractBackDayByNCycle(part_data,backNum,startNum,nan_method,7)


def getOneWeekdayFomExtractedData(extractData, weekday = 0):
    """
    在已经抽取的数据上取出周{weekday}的所有值，并去除时间一列,同时将前面的值转换成浮点
    :param extractData: 最后一列是时间
    :param weekday: 周几,0表示只去除时间一列
    :return:
    """
    if(weekday<0 or weekday>7):
        raise Exception("Parameter weekday error")
        return
    if weekday == 0:
        data = np.delete(extractData, extractData.shape[1] - 1, axis=1).astype(float)
    else:
        time = extractData.take(extractData.shape[1] - 1, axis=1)
        weekdays = pd.Series(time).map(getWeekday).values
        data = extractData[weekdays == weekday]
        data = np.delete(data, extractData.shape[1] - 1, axis=1).astype(float)
    return data

if __name__ == "__main__":

    import Parameter

    data = pd.read_csv(Parameter.meanfilteredAfterCompletion)
    part_data = data[data.shopid == 1]
    print part_data

    sameday = extractBackDay(part_data, 3, 0, nan_method_sameday_mean)
    print sameday
    print getOneWeekdayFomExtractedData(sameday, 5).shape
