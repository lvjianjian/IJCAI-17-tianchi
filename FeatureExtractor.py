# encoding=utf-8


import numpy as np
from DataRevision import getWeekday
import pandas as pd
import Parameter
nan_method_global_mean = "global_mean" #全局平均
nan_method_sameday_mean = "sameday_mean"#sameday平均

statistic_functon_mean = "statistic_functon_mean"
statistic_functon_median = "statistic_functon_median" #中位数
statistic_functon_max = "statistic_functon_max"
statistic_functon_min = "statistic_functon_min"
statistic_functon_std = "statistic_functon_std"

def getReplaceValue(replace, method, weekday):
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
    for i in range(skipNum, part_data.shape[0], 1):
        dataY.append(count[i])
        dataY.append(time[i])
    dataY = np.reshape(dataY, (part_data.shape[0] - skipNum, 2))
    return dataY

def extractWeekday(part_data, startNum = 0):
    """
    从{startNum}开始抽取weekday的值,
    :param part_data:需要含有两列组成，一列为count,另一列为time,注意这里的数据是要按日期补全的数据，不然会有不对应的情况
    :param startNum: 前面跳过的样例数，也就是从第{startNum}开始生成
    :return: ndarray,第一列为weekday
            第2列为哪个样例的日期
    """

    #先加入weekday一列，方便后面处理
    if "weekday" not in part_data.columns:
        part_data.insert(3, "weekday", part_data["time"].map(getWeekday))
    return part_data[startNum:][["weekday","time"]].values



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
        raise Exception("parameter startNum error")
        return
    if(backNum<1):
        raise Exception("parameter backNum error")
        return


    #先加入weekday一列，方便后面处理
    if "weekday" not in part_data.columns:
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

def extractBackWeekValue(part_data, backNum = 1, startNum = 0, nan_method=nan_method_global_mean,statistic_functon = None):
    """
    从{startNum}开始抽取前{backNum}周的sameday, 排列按照前一周的sameday，前第二周的sameday这样由近就及远排放
    :param part_data:需要含有两列组成，一列为count,另一列为time,注意这里的数据是要按日期补全的数据，不然会有不对应的情况
    :param backNum: 向前抽取多少周的周统计值
    :param startNum: 前面跳过的样例数，也就是从第{startNum}开始生成
    :param nan_method: 缺失值处理
    :param statistic_functon: 一周值的统计函数
    :return: ndarray,shape(处理的样例出数,backNum + 1),第一列到第{backNum}列为前第一周的周统计值到前第{backNum}周的周统计值
            第{backNum}+1列为哪个样例的日期
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
    if(statistic_functon is None):
        raise Exception("statistic_functon is None")
        return

    #先加入weekday一列，方便后面处理
    if "weekday" not in part_data.columns:
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
            current_wd = weekday[i]
            weekday_values=[]
            #拿出前一周的值
            for k in range(1, 8, 1):
                index = (i - 7 * (j + 1)) + (k - current_wd)
                if index < 0:
                    value = getReplaceValue(replace, nan_method, weekday[i])
                else:
                    value = count[int(index)]
                weekday_values.append(value)

            if statistic_functon == statistic_functon_mean:
                value = np.mean(weekday_values)
            elif statistic_functon == statistic_functon_median:
                value = np.median(weekday_values)
            elif statistic_functon == statistic_functon_max:
                value = np.max(weekday_values)
            elif statistic_functon == statistic_functon_min:
                value = np.min(weekday_values)
            elif statistic_functon == statistic_functon_std:
                value = np.std(weekday_values)
            dataX.append(value)
        dataX.append(time[i])
    dataX = np.reshape(dataX, (part_data.shape[0] - startNum, backNum + 1))
    return dataX

def extractWorkOrWeekend(part_data,startNum = 0):
    """
    从{startNum}开始抽取是工作日还是周末值,
    :param part_data:需要含有两列组成，一列为count,另一列为time,注意这里的数据是要按日期补全的数据，不然会有不对应的情况
    :param startNum: 前面跳过的样例数，也就是从第{startNum}开始生成
    :return: ndarray,第一列为workOrWeekend,第2列为哪个样例的日期,0为work,1为weekend
    """
    if(startNum <0 or startNum >= part_data.shape[0]):
        raise Exception("parameter skipNum error")
        return
    #先加入weekday一列，方便后面处理
    if "weekday" not in part_data.columns:
        part_data.insert(3, "weekday", part_data["time"].map(getWeekday))
    workOrWeekends = []
    weekds = part_data["weekday"].values.tolist()
    tims = part_data["time"].values.tolist()
    for i in range(startNum, part_data.shape[0], 1):
        weekd = weekds[i]
        workOrWeekends.append(1 if (weekd ==6 or weekd == 7) else 0)
        workOrWeekends.append(tims[i])
    workOrWeekends = np.reshape(workOrWeekends,(len(workOrWeekends)/2,2))
    return workOrWeekends


def extractWeatherInfo(part_data,startNum = 0,city_name = None):
    """
    从{startNum}开始抽取天气信息，晴或者多云或者小雨为0(转中雨不包含), 其他为1
    :param part_data:需要含有两列组成，一列为count,另一列为time,注意这里的数据是要按日期补全的数据，不然会有不对应的情况
    :param startNum: 前面跳过的样例数，也就是从第{startNum}开始生成
    :param city_name: 城市名字
    :return: ndarray,第一列为workOrWeekend,第2列为哪个样例的日期,0为work,1为weekend
    """
    if city_name is None:
        raise Exception("city name is None")
        return
    if(startNum <0 or startNum >= part_data.shape[0]):
        raise Exception("parameter skipNum error")
        return
    weathers = []
    times = part_data["time"].values
    weather_ = Parameter.getWeather_info()
    weather_part = weather_[weather_.area == city_name][["date", "weather"]]
    for i in range(startNum, part_data.shape[0], 1):
        time = times[i]
        v = weather_part[weather_part.date == time.replace("-", "/")]["weather"].values
        if(len(v) == 0 or pd.isnull(v[0])):
            v = 0
        else:
            v = v[0]
            if "云" in v or "晴" in v or ("小雨" in v and "阴" in v):
                v = 0
            else:
                v = 1
        weathers.append(v)
        weathers.append(time)

    weathers = np.reshape(weathers,(len(weathers)/2,2))
    return weathers

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


def onehot(data):
    """

    :param data: 某一列数据
    :return: encoder
    """
    from sklearn.preprocessing import OneHotEncoder
    temp = np.unique(data)
    temp2 = []
    for i in range(len(temp)):
        temp2.append([temp[i]])
    return OneHotEncoder(sparse=False).fit(temp2)


def labelEncoder(data):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder().fit(data)
    return encoder


if __name__ == "__main__":

    import Parameter

    data = pd.read_csv(Parameter.meanfilteredAfterCompletion)
    part_data = data[data.shopid == 2]
    # print part_data
    #
    # sameday = extractBackWeekValue(part_data, 3, 0, nan_method_sameday_mean, statistic_functon_mean)
    # print sameday
    # print getOneWeekdayFomExtractedData(sameday, 5).shape

    # weekdays = extractWeekday(part_data, 100)
    # print weekdays
    # weeks = extractWorkOrWeekend(part_data, 0)
    # w = getOneWeekdayFomExtractedData(weeks)
    # print onehot(w).transform([1]).toarray()
    print extractWeatherInfo(part_data, 0, "三明")
