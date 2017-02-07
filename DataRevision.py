#-*- coding=utf-8 -*-

""""
这个文件对客流量数据进行一些修复工作，如异常值使用均值代替，一些节假日的修正等
"""

import pandas as pd
from pandas import DataFrame,Series
from datetime import datetime


holidayInfo = pd.read_csv("data/holiday.csv", names=["time","holiday"])
holiday = holidayInfo["holiday"]
#转化为以日期为索引的Series
holidayInfo = Series(index=holidayInfo["time"])
holidayInfo[:] = holiday

def getWeekday(dateString):
    """
    根据日期判断星期几
    :param dateString:year-month-day,如：2016-10-21
    :return: 星期几（1到7），如：2016-10-21 返回 1
    """
    return datetime.strptime(dateString, "%Y-%m-%d").weekday() + 1


def getHoliday(dateString):
    """
    根据日期判断是否休息
    :param dateString: year-month-day,如：2016-10-21
    :return: 0代表工作日，-1代表休息日（这里休息日包含周末和节假日）
    """
    return holidayInfo[int(dateString.replace("-",""))]


def isHoliday(dataString):
    """
    判断是否休息
    :param dataString:
    :return: 休息返回true
    """
    return getHoliday(dataString) == -1

def isWeekend(dataString):
    """
    判断是否周末
    :param dataString:
    :return:  周末返回true
    """
    return getWeekday(dataString) == 6 or getWeekday(dataString) == 7

def revise1(x,weekday__mean,weekday__std):
    """
     1.对异常值用均值修正,如果当前值与平均值的差大于3倍方差时被均值代替
     2.对本应该休息却工作或者本应该工作却休息的值用均值代替
    :param x: 一行数据
    :param weekday__mean: 一周中每天的均值(即星期一道星期天的各自均值)
    :param weekday__std: 一周中每天的方差(即星期一道星期天的各自方差)
    :return:
    """
    weekday = x["weekday"]
    mean = weekday__mean.ix[weekday]["count"]
    std = weekday__std.ix[weekday]["count"]
    time_ = x["time"]
    holiday = getHoliday(time_)
    current = x["count"]
    if((isHoliday(time_) and not isWeekend(time_)) or (not isHoliday(time_) and isWeekend(time_))):
        #本应该休息却工作或者本应该工作却休息
        current = mean

    if(abs(current - mean)> 3 * std): #大于3倍方差了
        current = mean
    return current

if __name__ == "__main__":
    pay_info = pd.read_csv("data/user_pay_afterGrouping.csv")
    for i in range(1,2001,1):
        print i
        pay_part = pay_info[pay_info["shopid"] == i]
        pay_part.insert(3,"weekday",pay_part["time"].map(getWeekday))
        weekday__mean = pay_part[["count", "weekday"]].groupby("weekday").mean()
        weekday__std = pay_part[["count", "weekday"]].groupby("weekday").std()
        pay_info.loc[pay_info["shopid"] == i,"count"] = pay_part.apply(lambda x:revise1(x,weekday__mean,weekday__std),axis=1)
    pay_info.to_csv("data/user_pay_afterGroupingAndRevision.csv")