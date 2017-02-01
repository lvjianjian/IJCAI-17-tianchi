#-*- coding=gbk -*-

import pandas as pd
import time
import  numpy as np
import matplotlib.pyplot as plt
from  matplotlib.dates  import datestr2num
from  matplotlib.dates  import num2date,datestr2num,drange,DateFormatter
import datetime
import re
import function_collection as fc
from pylab import mpl
# map(num2date, map(datestr2num, all_view_data[all_view_data.shopid == 2]['time'].tolist()))
def numDatetoStr1(DT):
     '''
     :param DT: input datetime
     :return: converted string;fomat:'%Y\%m\%d'
     2015/3/2 not 2015/03/02
     '''
     temp_date=str(num2date(DT)).split(' ')[0]
     words=temp_date.split('-')
     if words[1][0]=='0':
         temp_str=words[1][1]
         words[1]=temp_str
     if words[2][0]=='0':
         temp_str=words[2][1]
         words[2]=temp_str
     return words[0]+'/'+words[1]+'/'+words[2]


def numDatetoStr2(DT):
    '''
    :param DT: input datetime
    :return: converted string;fomat:'%Y-%m-%d'
    2015-3-2 not 2015-03-02
    '''
    temp_date = str(num2date(DT)).split(' ')[0]
    words = temp_date.split('-')
    if words[1][0] == '0':
        temp_str = words[1][1]
        words[1] = temp_str
    if words[2][0] == '0':
        temp_str = words[2][1]
        words[2] = temp_str
    return words[0] + '-' + words[1] + '-' + words[2]

def StrToDate_1(datestr):
    '''

    :param datestr: %Y/%m/%d
    :return: date:datetime
    '''
    return datetime.datetime.strptime(datestr,'%Y/%m/%d')

def DateToStr_1(date):
    '''

    :param date: datetime
    :return: datestr: %Y-%m-%d
    '''
    return datetime.datetime.strftime(date,'%Y-%m-%d')

def StrDate1ToStrDate(datestr):
    '''

    :param datestr: %Y/%m/%d
    :return: datestr:%Y-%m-%d
    '''
    return DateToStr_1(StrToDate_1(datestr))
weather_path='data/weather_info.csv'
pay_path='data/user_pay_afterGrouping.csv'
shop_path='data/shop_info.txt'
weather_info=pd.read_csv(weather_path,encoding='gb2312')
pay_info=pd.read_csv(pay_path)
shop_info=pd.read_csv(shop_path,names=['shopid','city_name','location_id','per_pay','score','comment_cnt','shop_level','cate1_name','cate2_name','cate3_name'])
s='三明'.decode('gbk')
startDate=datetime.datetime(2015,6,1)
endDate=datetime.datetime(2017,1,1)
delta=datetime.timedelta(days=20)
xdates= drange(startDate , endDate, delta)
delta1=datetime.timedelta(days=1)
dlist=drange(startDate,endDate,delta1)
# datet = datetime.datetime.strptime('2017-1-29','%Y-%m-%d')
# print datet.isoweekday()
dates = map(numDatetoStr1,dlist)
# print dates
city_drange_weather_date=weather_info[(weather_info['area']==s) & (weather_info['date'].isin(dates)) &  (weather_info['weather_level']==3)].date # datestr:%Y/%m/%d
weather_date_list = map(StrDate1ToStrDate,city_drange_weather_date.values)

fig=plt.figure(figsize=(10,8))
plt.xlabel('date')
plt.ylabel('counts')
plt.xticks(xdates)
f=[1,2]
# obpay=pay_info[(pay_info['shopid']==1) & (pay_info['time'].isin(weather_date_list))]
obpay=pay_info[pay_info.index.isin(f)]
dateindex=obpay.time.values
print dateindex
countslist=obpay.loc[:,'count'].values
view_ax=fig.add_subplot(1,1,1)
view_ax.set_xticklabels(xdates, rotation=45, size=5)
view_ax.plot_date(dateindex, countslist,color='g', marker='*');
fig.show()
time.sleep(10000)
# pay_index=pay_info[]


# print weather_info[(weather_info['area']==s ) & (weather_info['date'].isin(dates))]







