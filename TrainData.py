# encoding=utf-8

import pandas as pd
import time
import  numpy as np
import matplotlib.pyplot as plt
from  matplotlib.dates  import datestr2num
from  matplotlib.dates  import num2date,datestr2num,drange,DateFormatter
import datetime
from  datetime import  datetime as ddt
import re
import function_collection as fc
from pylab import mpl
fc.set_ch()
def RemoveDatezero(datestr):
    words=datestr.split('/')
    if words[1][0]=='0':
        words[1]=words[1][1]
    if words[2][0]=='0':
        words[2]=words[2][1]
    return words[0]+'/'+words[1]+'/'+words[2]

def transferNum(level):
    if level==-3 or (level==0 or level==3):
        return 0
    if level==-2 or (level==1 or level==4):
        return 1
    if level==-1 or (level==2 or level==5):
        return 2
    return -1
namelist=['weekday','day_1','day_2','day_3','day_4','day_5','day_6','day_7','same_day',
          'day_1_weather','day_2_weather','day_3_weather','day_4_weather','day_5_weather','day_6_weather',
          'day_7_weather','same_day_weather']


# train_data=pd.DataFrame(names=namelist)
# print train_data
shop_path='data/shop_info.txt'
shop_info=pd.read_csv(shop_path,names=['shop_id','city_name','location_id','per_pay','score',
                                       'comment_cnt','shop_level','cate1_name','cate2_name','cate3_name'])


shop_id_Series=shop_info['shop_id'].values
user_pay_path='data/user_pay_afterGrouping.csv'
user_view_path='data/user_view_afterGrouping.csv'
weather_path='data/weather_info.csv'
weather_info=pd.read_csv(weather_path,encoding='gb2312')
# print weather_info.head()
# print type(weather_info[weather_info['area']=='三明'].weather_level.values[0])
user_pay_info=pd.read_csv(user_pay_path)
user_view_info=pd.read_csv(user_view_path)
# Datatime_Series = pd.DatetimeIndex(['2015/2/1', '2016/2/1'])
# print type(Datatime_Series[0])
# print datetime.datetime.strftime(Datatime_Series[1]-datetime.timedelta(days=365),'%Y/%m/%d')
# time.timed
# print map(datetime.datetime.isoweekday,Datatime_Series)
# print datetime.timedelta(days=365)==Datatime_Series[1]-Datatime_Series[0]


for i,shop_id in enumerate(shop_id_Series):
    weekday = []
    pay_day_1 = []
    pay_day_2 = []
    pay_day_3 = []
    pay_day_4 = []
    pay_day_5 = []
    pay_day_6 = []
    pay_day_7 = []
    pay_same_day = []

    view_day_1 = []
    view_day_2 = []
    view_day_3 = []
    view_day_4 = []
    view_day_5 = []
    view_day_6 = []
    view_day_7 = []

    day_1_weather = []
    day_2_weather = []
    day_3_weather = []
    day_4_weather = []
    day_5_weather = []
    day_6_weather = []
    day_7_weather = []
    same_day_weather = []
    labellist = []

    print 'shopId:',shop_id
    cate1=str(shop_info[shop_info.shop_id==shop_id].cate1_name.values[0])
    # if shop_id<1636:
    #     continue
    if cate1=='美食':
        father_path='food_csvfile1\\'
    else:
        if cate1=='超市便利店':
           father_path = 'supermarket_csvfile1\\'
        else:
            father_path = 'other_csvfile1\\'

    pay_time_list=user_pay_info[user_pay_info['shopid']==shop_id]['time'].tolist()
    pay_count_list=user_pay_info[user_pay_info['shopid']==shop_id]['count'].tolist()
    view_time_list = user_view_info[user_view_info['shopid'] == shop_id]['time'].tolist()
    view_count_list = user_view_info[user_view_info['shopid'] == shop_id]['count'].tolist()

    city=shop_info[shop_info['shop_id']==shop_id].city_name.values[0]
    city=city.decode('utf-8')
    print 'City', city
    # print 'city:',city
    city_weather_info=weather_info[weather_info.area == city]
    # print 'city_weather', city_weather_info.head()
    cur_pay_series = pd.Series(pay_count_list, index=pay_time_list)
    cur_view_series = pd.Series(view_count_list, index=view_time_list)
    #  重采样
    pay_Datatime_Series=pd.DatetimeIndex([pay_time_list[0],pay_time_list[len(pay_time_list)-1]])
    if len(view_time_list)!=0:
      view_Datatime_Series=pd.DatetimeIndex([view_time_list[0],view_time_list[len(view_time_list)-1]])
    else:
      view_Datatime_Series = pd.DatetimeIndex([pay_time_list[0], pay_time_list[len(pay_time_list) - 1]])

    pay_tem_series = pd.Series([0, 0], index=pay_Datatime_Series).resample('D', ).pad()
    view_tem_series = pd.Series([0, 0], index=view_Datatime_Series).resample('D', ).pad()

    for time_item in pay_time_list:
        pay_tem_series[time_item]=cur_pay_series[time_item]
    for time_item in view_time_list:
        view_tem_series[time_item]=cur_view_series[time_item]

    for time_item in pay_time_list:
        dateDay=ddt.strptime(time_item,'%Y/%m/%d')
        cur_count=pay_tem_series[time_item]
        cur_weekday=ddt.isoweekday(dateDay)


        week7_date = dateDay - datetime.timedelta(days=(cur_weekday + 0))
        week6_date = dateDay - datetime.timedelta(days=(cur_weekday + 1))
        week5_date = dateDay - datetime.timedelta(days=(cur_weekday + 2))
        week4_date = dateDay - datetime.timedelta(days=(cur_weekday + 3))
        week3_date = dateDay - datetime.timedelta(days=(cur_weekday + 4))
        week2_date = dateDay - datetime.timedelta(days=(cur_weekday + 5))
        week1_date = dateDay - datetime.timedelta(days=(cur_weekday + 6))
        same_date =  dateDay - datetime.timedelta(days=14)

        week7_datestr = ddt.strftime(week7_date, '%Y/%m/%d')
        week6_datestr = ddt.strftime(week6_date, '%Y/%m/%d')
        week5_datestr = ddt.strftime(week5_date, '%Y/%m/%d')
        week4_datestr = ddt.strftime(week4_date, '%Y/%m/%d')
        week3_datestr = ddt.strftime(week3_date, '%Y/%m/%d')
        week2_datestr = ddt.strftime(week2_date, '%Y/%m/%d')
        week1_datestr = ddt.strftime(week1_date, '%Y/%m/%d')
        same_datestr = ddt.strftime(same_date, '%Y/%m/%d')

#################################################################################################################
        if week7_date>=pay_Datatime_Series[0]:
            pay_week7_count = pay_tem_series[RemoveDatezero(week7_datestr)]
        else:
            pay_week7_count=np.nan

        if week6_date >= pay_Datatime_Series[0]:
            pay_week6_count = pay_tem_series[RemoveDatezero(week6_datestr)]
        else:
            pay_week6_count= np.nan

        if week5_date >= pay_Datatime_Series[0]:
            pay_week5_count = pay_tem_series[RemoveDatezero(week5_datestr)]
        else:
            pay_week5_count = np.nan

        if week4_date >= pay_Datatime_Series[0]:
            pay_week4_count = pay_tem_series[RemoveDatezero(week4_datestr)]
        else:
            pay_week4_count = np.nan

        if week3_date >= pay_Datatime_Series[0]:
            pay_week3_count = pay_tem_series[RemoveDatezero(week3_datestr)]
        else:
            pay_week3_count = np.nan

        if week2_date >= pay_Datatime_Series[0]:
            pay_week2_count = pay_tem_series[RemoveDatezero(week2_datestr)]
        else:
            pay_week2_count = np.nan

        if week1_date >= pay_Datatime_Series[0]:
            pay_week1_count = pay_tem_series[RemoveDatezero(week1_datestr)]
        else:
            pay_week1_count = np.nan

        if same_date >= pay_Datatime_Series[0]:
            pay_same_count = pay_tem_series[RemoveDatezero(same_datestr)]
        else:
            pay_same_count = np.nan


############################################################################################################################

        if week7_date >= view_Datatime_Series[0] and week7_date <= view_Datatime_Series[1]:
            view_week7_count = view_tem_series[RemoveDatezero(week7_datestr)]
        else:
            view_week7_count = np.nan

        if week6_date >= view_Datatime_Series[0]  and week6_date <= view_Datatime_Series[1]:
            view_week6_count = view_tem_series[RemoveDatezero(week6_datestr)]
        else:
            view_week6_count = np.nan

        if week5_date >= view_Datatime_Series[0]  and week5_date <= view_Datatime_Series[1]:
            view_week5_count = view_tem_series[RemoveDatezero(week5_datestr)]
        else:
            view_week5_count = np.nan

        if week4_date >= view_Datatime_Series[0]  and week4_date <= view_Datatime_Series[1]:
            view_week4_count = view_tem_series[RemoveDatezero(week4_datestr)]
        else:
            view_week4_count = np.nan

        if week3_date >= view_Datatime_Series[0]  and week3_date <= view_Datatime_Series[1]:
            view_week3_count = view_tem_series[RemoveDatezero(week3_datestr)]
        else:
            view_week3_count = np.nan

        if week2_date >= view_Datatime_Series[0]  and week2_date <= view_Datatime_Series[1]:
            view_week2_count = view_tem_series[RemoveDatezero(week2_datestr)]
        else:
            view_week2_count = np.nan

        if week1_date >= view_Datatime_Series[0]  and week1_date <= view_Datatime_Series[1]:
            view_week1_count = view_tem_series[RemoveDatezero(week1_datestr)]
        else:
            view_week1_count = np.nan

        weekday.append(cur_weekday)
        labellist.append(cur_count)

        # print 'week4_datestr',RemoveDatezero(week4_datestr)

        week7_weather=map(transferNum,city_weather_info[city_weather_info['date']==RemoveDatezero(week7_datestr)].weather_level.values)
        week6_weather=map(transferNum,city_weather_info[city_weather_info['date']==RemoveDatezero(week6_datestr)].weather_level.values)
        week5_weather=map(transferNum,city_weather_info[city_weather_info['date']==RemoveDatezero(week5_datestr)].weather_level.values)
        week4_weather=map(transferNum,city_weather_info[city_weather_info['date']==RemoveDatezero(week4_datestr)].weather_level.values)
        week3_weather=map(transferNum,city_weather_info[city_weather_info['date']==RemoveDatezero(week3_datestr)].weather_level.values)
        week2_weather=map(transferNum,city_weather_info[city_weather_info['date']==RemoveDatezero(week2_datestr)].weather_level.values)
        week1_weather=map(transferNum,city_weather_info[city_weather_info['date']==RemoveDatezero(week1_datestr)].weather_level.values)
        sameday_weather=map(transferNum,city_weather_info[city_weather_info['date']==RemoveDatezero(same_datestr)].weather_level.values)
        if len(week7_weather)==0:
            week7_weather=np.nan
        else:
            week7_weather=week7_weather[0]
        if len(week6_weather) == 0:
            week6_weather = np.nan
        else:
            week6_weather = week6_weather[0]
        if len(week5_weather) == 0:
            week5_weather = np.nan
        else:
            week5_weather = week5_weather[0]
        if len(week4_weather) == 0:
            week4_weather = np.nan
        else:
            week4_weather = week4_weather[0]
        if len(week3_weather) == 0:
            week3_weather = np.nan
        else:
            week3_weather = week3_weather[0]

        if len(week2_weather) == 0:
            week2_weather = np.nan
        else:
            week2_weather = week2_weather[0]

        if len(week1_weather) == 0:
            week1_weather = np.nan
        else:
            week1_weather = week1_weather[0]

        if len(sameday_weather) == 0:
            sameday_weather = np.nan
        else:
            sameday_weather = sameday_weather[0]

#########################################################################################################################

        pay_day_1.append(pay_week1_count)
        pay_day_2.append(pay_week2_count)
        pay_day_3.append(pay_week3_count)
        pay_day_4.append(pay_week4_count)
        pay_day_5.append(pay_week5_count)
        pay_day_6.append(pay_week6_count)
        pay_day_7.append(pay_week7_count)
        pay_same_day.append(pay_same_count)

        view_day_1.append(view_week1_count)
        view_day_2.append(view_week2_count)
        view_day_3.append(view_week3_count)
        view_day_4.append(view_week4_count)
        view_day_5.append(view_week5_count)
        view_day_6.append(view_week6_count)
        view_day_7.append(view_week7_count)



        day_1_weather.append(week1_weather)
        day_2_weather.append(week2_weather)
        day_3_weather.append(week3_weather)
        day_4_weather.append(week4_weather)
        day_5_weather.append(week5_weather)
        day_6_weather.append(week6_weather)
        day_7_weather.append(week7_weather)
        same_day_weather.append(sameday_weather)
    trainSet={
              'weekday':weekday,
              'pay_day1':pay_day_1,
              'pay_day2':pay_day_2,
              'pay_day3':pay_day_3,
              'pay_day4':pay_day_4,
              'pay_day5':pay_day_5,
              'pay_day6':pay_day_6,
              'pay_day7':pay_day_7,
              'same_day':pay_same_day,
              'day1_weather':day_1_weather,
              'day2_weather':day_2_weather,
              'day3_weather':day_3_weather,
              'day4_weather':day_4_weather,
              'day5_weather':day_5_weather,
              'day6_weather':day_6_weather,
              'day7_weather':day_7_weather,
              'sameday_weather':same_day_weather,
              'view_day1':view_day_1,
              'view_day2':view_day_2,
              'view_day3':view_day_3,
              'view_day4':view_day_4,
              'view_day5':view_day_5,
              'view_day6':view_day_6,
              'view_day7':view_day_7,
              'count':labellist
     }

    del weekday
    del pay_day_1
    del pay_day_2
    del pay_day_3
    del pay_day_4
    del pay_day_5
    del pay_day_6
    del pay_day_7
    del pay_same_day

    del view_day_1
    del view_day_2
    del view_day_3
    del view_day_4
    del view_day_5
    del view_day_6
    del view_day_7

    del day_1_weather
    del day_2_weather
    del day_3_weather
    del day_4_weather
    del day_5_weather
    del day_6_weather
    del day_7_weather
    del same_day_weather
    del labellist
    cur_path=father_path+str(shop_id)+'_trainset.csv'
    cur_df=pd.DataFrame(trainSet)
    cur_df.to_csv(cur_path,index=True)

