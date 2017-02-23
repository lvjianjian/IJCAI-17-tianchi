#-*- coding=utf-8 -*-

""""
这个文件对客流量数据进行一些修复工作，如异常值使用均值代替，一些节假日的修正等
"""

import pandas as pd
from pandas import DataFrame,Series
import datetime
import Parameter
import numpy as np

holidayInfo = pd.read_csv(Parameter.holidayPath, names=["time","holiday"])
holiday = holidayInfo["holiday"]
#转化为以日期为索引的Series
holidayInfo = Series(index=holidayInfo["time"])
holidayInfo[:] = holiday

outlier_remove = "outlier_remove"
outlier_sameday_replace="outlier_sameday_replace"
outlier_day_replace="outlier_day_replace"
succession_completion = "succession_completion"
succession_turncate = "succession_turncate"

def getWeekday(dateString):
    """
    根据日期判断星期几
    :param dateString:year-month-day,如：2016-10-21
    :return: 星期几（1到7），如：2016-10-21 返回 1
    """
    return datetime.datetime.strptime(dateString, "%Y-%m-%d").weekday() + 1


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
     1.对异常值用均值修正,如果当前值与平均值的差大于3倍标准差时被均值代替
     2.对本应该休息却工作或者本应该工作却休息的值用均值代替
    :param x: 一行数据
    :param weekday__mean: 一周中每天的均值(即星期一道星期天的各自均值)
    :param weekday__std: 一周中每天的标准差(即星期一道星期天的各自标准差)
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

    if(abs(current - mean)> 3 * std): #大于3倍标准差了
        current = mean
    return current

def revise2(oneshop_pay_info,weekday__mean):
    """
     3.对一些没有值的天也用均值代替
    :param oneshop_pay_info: 一家商店的所有数据
    :param weekday__mean: 一周中每天的均值(即星期一道星期天的各自均值)
    :return:
    """
    data_index = pd.DatetimeIndex([oneshop_pay_info["time"].values[0], oneshop_pay_info["time"].values[oneshop_pay_info.shape[0]-1]])
    completeData = Series(oneshop_pay_info["count"].values,index=pd.DatetimeIndex(oneshop_pay_info["time"].values)).resample("D").asfreq()
    # print completeData
    nan_values = completeData[pd.isnull(completeData.values)]
    for date in nan_values.index:
        weekday = getWeekday(date.strftime("%Y-%m-%d"))
        completeData[date] = weekday__mean.ix[weekday]
    return completeData

def reviseAll(data, saveFile, completion = False):
    newData = pd.DataFrame(columns=data.columns[0:])

    for i in range(1, 2001, 1):
        print i
        pay_part = data[data["shopid"] == i]
        pay_part.insert(3,"weekday",pay_part["time"].map(getWeekday))
        weekday__groupby = pay_part[["count", "weekday"]].groupby("weekday")
        weekday__mean = weekday__groupby.mean()
        weekday__median = weekday__groupby.median()
        weekday__std = weekday__groupby.std()
        part_apply_revised = pay_part.apply(lambda x: revise1(x, weekday__mean, weekday__std), axis=1)
        data.loc[data["shopid"] == i, "count"] = part_apply_revised
        pay_part = data[data["shopid"] == i]
        if(completion):
            completeData = revise2(pay_part, weekday__mean)
            frame = pd.DataFrame({"shopid":i,
                                  "time":completeData.index,
                                  "count":completeData.values})
        else:
            frame = pd.DataFrame({"shopid":i,
                                    "time":pay_part["time"].values,
                                    "count":pay_part["count"].values})
        frames=[newData,frame]
        newData = pd.concat(frames,ignore_index=True)
    if completion:
        newData["shopid"] = newData["shopid"].astype(int)
    newData.to_csv(saveFile)


def getReplaceCount(part,method,window,currentindex):
    if method == outlier_remove:
        return -1

    elif method == outlier_day_replace:
        preindex = (currentindex - window) if (currentindex-window)>=0 else 0
        postindex = (currentindex+window) if (currentindex+window)<part.shape[0] else part.shape[0] - 1
        return part[preindex:postindex, 2].mean()

    elif method == outlier_sameday_replace:
        times = part[:,1].tolist()
        currenttime = part[currentindex][1]
        # print currenttime
        import datetime
        format="%Y-%m-%d"
        timedelta = datetime.timedelta(7)
        strptime = datetime.datetime.strptime(currenttime, format)
        counts=[]
        for k in range(window):
            pretime = strptime - (k+1)*timedelta
            pretime_s = pretime.strftime(format)
            try:
                index = times.index(pretime_s)
            except:
                index = -1
            if index != -1:
                counts.append(part[index][2])

            posttime = strptime + (k+1)*timedelta
            posttime_s = posttime.strftime(format)
            try:
                index = times.index(posttime_s)
            except:
                index = -1
            if index != -1:
                counts.append(part[index][2])
        # import numpy as np
        # print "%d would be replace as %d" % (part[currentindex][2], np.mean(counts))
        if len(counts) != 0:
            import numpy as np
            return np.mean(counts)
        else:
            return -1

    else:
        raise Exception("no replace method %s" % method)

def reviseOneShop(part,windowRadious,method,replace_day_window,replace_sameday_window,multi,checkHoliday,needRevise):
    """
    :param part:
    :param windowRadious:
    :param method:
    :param replace_day_window:
    :param replace_sameday_window:
    :param multi:
    :param checkHoliday:
    :param needRevise
    :return: [count_list,time_list]
    """
    time=[]
    count=[]
    part = part.values
    length = part.shape[0]
    if method == outlier_sameday_replace:
        window = replace_sameday_window
    elif method == outlier_day_replace:
        window = replace_day_window
    else:
        window = -1

    for i in range(length):
        current_time = part[i][1]
        current_count = part[i][2]
        #不需要修复直接放入
        if not needRevise:
            count.append(current_count)
            time.append(current_time)
            continue
        # current_weekday = part[i][3]
        #计算以当前天周围windowRadious的均值和方差
        pre_index = (i - windowRadious) if (i-windowRadious) >= 0 else 0
        post_index = (i + windowRadious) if(i + windowRadious) < length else length-1
        window_data = (part[pre_index:post_index, 2])
        mean = window_data.mean()
        std = window_data.std()
        if(abs(current_count - mean) > multi * std):
            # print "std"
            current_count = getReplaceCount(part, method, window, i)
        elif((isHoliday(current_time) and not isWeekend(current_time)) or (not isHoliday(current_time) and isWeekend(current_time))):
        #本应该休息却工作或者本应该工作却休息
            if checkHoliday:
                # print "holiday"
                current_count = getReplaceCount(part, method, window, i)

        if(current_count != -1):
            count.append(current_count)
            time.append(current_time)
    # print len(count)
    return [count, time]


def turancate(part_data):
    count = []
    time = []
    part_data.insert(3,"weekday",part_data["time"].map(getWeekday))
    weekday = part_data["weekday"].values
    oldtime = part_data["time"].values
    oldcount=part_data["count"].values
    start = weekday[0]
    for i in range(len(weekday)):
        if (weekday[i] == start):
            count.append(oldcount[i])
            time.append(oldtime[i])
            start = start % 7 + 1
    return [count, time]

def reviseAll2(data,saveFilePath, windowRadious=30, method=outlier_sameday_replace, replace_day_window=7, replace_sameday_window=2, multi = 3, checkHoliday=True,succession = None, needRevise = True):
    """
    :param data:
    :param saveFilePath:
    :param windowRadious: 窗口半径大小
    :param method: 对于异常值的替换方法
    :param replace_day_window: 对于outlier_day_replace方式时，这个参数指定前后获取day的个数，默认7
    :param replace_sameday_window: 对于outlier_sameday_replace方式时，这个参数指定前后获取sameday的个数，默认为1
    :param multi:=3,大于{multi}倍方差时检测异常
    :param checkHoliday: 检查节假日
    :param succession:day连续化，None为不连续化，succession_turncate为截断法连续化，completion为补全法连续化
    :param needRevise:是否需要修正,默认为True
    :return:
    """
    if data.columns[0] != "shopid":
        raise Exception("first column should be shopid")
        return
    if data.columns[1] != "time":
        raise Exception("first column should be shopid")
        return
    if data.columns[2] != "count":
        raise Exception("first column should be shopid")
        return
    import pandas as pd
    newData = pd.DataFrame(columns=data.columns[0:])
    for j in range(1, 2001, 1):
        print j
        pay_part = data[data["shopid"] == j]
        # pay_part.insert(3, "weekday", pay_part["time"].map(getWeekday))
        count, time = reviseOneShop(pay_part,windowRadious,method,replace_day_window,replace_sameday_window,multi,checkHoliday,needRevise)

        frame = pd.DataFrame({"shopid": j, "time": time, "count": count})


        if(succession is None):
           pass
        elif(succession == succession_completion):
            frame.insert(3,"weekday",frame["time"].map(getWeekday))
            weekday__groupby = frame[["count", "weekday"]].groupby("weekday")
            weekday__mean = weekday__groupby.mean()
            completeData = revise2(frame, weekday__mean)
            frame = pd.DataFrame({"shopid":j,
                              "time":completeData.index,
                              "count":completeData.values})
        elif(succession == succession_turncate):
            count, time = turancate(frame)
            frame = pd.DataFrame({"shopid": j, "time": time, "count": count})
        else:
            raise Exception("parameter succession error")
            return
        newData = pd.concat([newData, frame], ignore_index=True)

    newData['shopid'] = newData['shopid'].apply(int)
    newData.to_csv(saveFilePath)

if __name__ == "__main__":
    data = pd.read_csv("data/user_pay_afterGrouping.csv")
    reviseAll2(data, "data/user_pay_afterGroupingAndTurncate.csv", succession=succession_turncate,needRevise=False)