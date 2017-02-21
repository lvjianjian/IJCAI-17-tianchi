# encoding=utf-8
import pandas as pd
import time
import  matplotlib as mplt
mplt.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from  matplotlib.dates  import datestr2num
from  matplotlib.dates  import num2date,datestr2num,drange,DateFormatter
import datetime
from datetime import datetime as ddt
import numpy as np
import  Parameter as para
import  function_collection as fc
import datetools
import sys
import  function_collection as fc

reload(sys)
sys.setdefaultencoding("utf-8")
from pylab import mpl
fc.set_ch()
'''
此文件代码用于分析店家的销售模式
'''



def countByDate(groupingData,holiday_data,picturepath):
    '''
    根据日期来统计所有店家对时间的敏感
    :param groupingData:
    :return:
    '''
    startDate = datetime.datetime(2015, 7, 1)
    endDate = datetime.datetime(2016, 11,1)
    delta = datetime.timedelta(days=1)
    Datelist = map(datetools.numDatetoStr_removezeros, drange(startDate, endDate, delta)) #strformat:%Y/%m/%d
    holidaydateList = holiday_data[   (holiday_data['flag']==-1)
                                    & (holiday_data['time']>=20150701)
                                    & (holiday_data['time']<20161101)]['time'].values
    holidaydateList=map(datetools.DateStr_3ToDatestr_1,holidaydateList) #strformat:%Y/%m/%d 2015/03/02
    holidaydateList=map(datetools.Datestr1_removezeros,holidaydateList) #strformat:%Y/%m/%d 2015/3/2
    countlist=[]
    holidaycountlist=[]
    for datestr in Datelist:
        cur_list=groupingData[groupingData['time']==datestr]['count'].values
        cur_sum = 0
        if len(cur_list)!=0:
            cur_sum=np.sum(cur_list)
        if datestr in holidaydateList:
            holidaycountlist.append(cur_sum)
        countlist.append(cur_sum)
    # print len(holidaydateList)
    plot_series=pd.Series(countlist,index=Datelist)
    plot_series_holiday=pd.Series(holidaycountlist,index=holidaydateList)
  ###################数据设置完毕，开始画图#####################################
    plot_delta=datetime.timedelta(days=20)
    dates = drange(startDate, endDate, plot_delta)
    fig = plt.figure(figsize=(15, 8))
    plt.xlabel('date')
    plt.ylabel('counts')
    plt.grid(True)

    view_ax = fig.add_subplot(1, 1, 1)
    view_ax.set_xticklabels(dates, rotation=45, size=10)
    view_ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    view_ax.plot_date(plot_series.index, plot_series.values, 'm-', marker='.', linewidth=0.5)
    view_ax.plot_date(plot_series_holiday.index, plot_series_holiday.values, color='r', marker='.')
    plt.savefig(picturepath)
    view_ax.clear()


def countByDateAndCate2(groupingData,holiday_data,shop_data,picturepath):
    '''
    根据日期来统计所有店家对时间的敏感
    :param groupingData:
    :return:
    '''
    cate2_list=shop_data['cate2_name'].unique()
    startDate = datetime.datetime(2015, 07, 1)
    endDate = datetime.datetime(2016, 11, 1)
    delta = datetime.timedelta(days=1)
    Datelist = map(datetools.numDatetoStr_removezeros, drange(startDate, endDate, delta))
    holidaydateList = holiday_data[(holiday_data['flag'] == -1)
                                   & (holiday_data['time'] >= 20150701)
                                   & (holiday_data['time'] < 20161101)]['time'].values  # strformat:%Y%m%d 20150302
    holidaydateList = map(datetools.DateStr_3ToDatestr_1, holidaydateList)  # strformat:%Y/%m/%d 2015/03/02
    holidaydateList = map(datetools.Datestr1_removezeros, holidaydateList)  # strformat:%Y/%m/%d 2015/3/2
    for i,cate in enumerate(cate2_list):
        print cate
        countlist=[]
        holidaycountlist = []
        shopidlist=shop_data[shop_data['cate2_name']==cate]['shopid'].values
        shop_Num=len(shopidlist)
        # print shopidlist
        for datestr in Datelist:
            cur_list=groupingData[(groupingData['time']==datestr) &
                                  (groupingData['shopid'].isin(shopidlist))]['count'].values
            # print cur_list
            cur_sum = 0
            if len(cur_list)!=0:
                cur_sum=np.sum(cur_list)

            #当前日期是假期的话就将当前值加入到假期序列中
            if datestr in holidaydateList:
                holidaycountlist.append(cur_sum)

            countlist.append(cur_sum)
        plot_series=pd.Series(countlist,index=Datelist)
        plot_series_holiday = pd.Series(holidaycountlist, index=holidaydateList)
        ###################数据设置完毕，开始画图#####################################
        plot_delta=datetime.timedelta(days=20)
        dates = drange(startDate, endDate, plot_delta)
        fig = plt.figure(figsize=(15, 8))
        plt.xlabel('date')
        plt.ylabel('count')
        plt.title('leve2_cate:'+cate+'\n商家数:'+str(shop_Num))
        plt.grid(True)
        view_ax = fig.add_subplot(1, 1, 1)
        view_ax.set_xticklabels(dates, rotation=45, size=10)
        view_ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        view_ax.plot_date(plot_series.index, plot_series.values, 'm-', marker='.', linewidth=0.5)
        view_ax.plot_date(plot_series_holiday.index, plot_series_holiday.values, color='r', marker='.')
        plt.savefig(picturepath+str(i)+'_trend.png')
        view_ax.clear()
        plt.close()

def countByDateAndCate1(groupingData,holiday_data,shop_data,picturepath):
    '''
    根据日期来统计所有店家对时间的敏感
    :param groupingData:
    :return:
    '''
    cate1_list=shop_data['cate1_name'].unique()
    startDate = datetime.datetime(2015,07, 1)
    endDate = datetime.datetime(2016, 11,1)
    delta = datetime.timedelta(days=1)
    Datelist = map(datetools.numDatetoStr_removezeros, drange(startDate, endDate, delta))
    holidaydateList = holiday_data[(holiday_data['flag'] == -1)
                                   & (holiday_data['time'] >= 20150701)
                                   & (holiday_data['time'] < 20161101)]['time'].values  # strformat:%Y%m%d 20150302
    holidaydateList = map(datetools.DateStr_3ToDatestr_1, holidaydateList)  # strformat:%Y/%m/%d 2015/03/02
    holidaydateList = map(datetools.Datestr1_removezeros, holidaydateList)  # strformat:%Y/%m/%d 2015/3/2
    for i,cate in enumerate(cate1_list):
        print cate
        countlist=[]
        holidaycountlist = []
        shopidlist=shop_data[shop_data['cate1_name']==cate]['shopid'].values
        shop_Num=len(shopidlist)
        # print shopidlist
        for datestr in Datelist:
            cur_list=groupingData[(groupingData['time']==datestr) &
                                  (groupingData['shopid'].isin(shopidlist))]['count'].values
            # print cur_list
            cur_sum = 0
            if len(cur_list)!=0:
                cur_sum=np.sum(cur_list)

            #当前日期是假期的话就将当前值加入到假期序列中
            if datestr in holidaydateList:
                holidaycountlist.append(cur_sum)

            countlist.append(cur_sum)
        plot_series=pd.Series(countlist,index=Datelist)
        plot_series_holiday = pd.Series(holidaycountlist, index=holidaydateList)
        ###################数据设置完毕，开始画图#####################################
        plot_delta=datetime.timedelta(days=20)
        dates = drange(startDate, endDate, plot_delta)
        fig = plt.figure(figsize=(15, 8))
        plt.xlabel('date')
        plt.ylabel('count')
        plt.title('level1_cate:'+cate+'\n商家数:'+str(shop_Num))
        plt.grid(True)
        view_ax = fig.add_subplot(1, 1, 1)
        view_ax.set_xticklabels(dates, rotation=45, size=10)
        view_ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        view_ax.plot_date(plot_series.index, plot_series.values, 'm-', marker='.', linewidth=0.5)
        view_ax.plot_date(plot_series_holiday.index, plot_series_holiday.values, color='r', marker='.')
        plt.savefig(picturepath+str(i)+'_trend.png')
        view_ax.clear()
        plt.close()

def countByDateAndCate3(groupingData,holiday_data,shop_data,picturepath):
    '''
    根据日期来统计所有店家对时间的敏感
    :param groupingData:
    :return:
    '''
    cate3_list=shop_data['cate3_name'].unique()
    startDate = datetime.datetime(2015, 07, 1)
    endDate = datetime.datetime(2016, 11,1)
    delta = datetime.timedelta(days=1)
    Datelist = map(datetools.numDatetoStr_removezeros, drange(startDate, endDate, delta))
    holidaydateList = holiday_data[(holiday_data['flag'] == -1)
                                   & (holiday_data['time'] >= 20150701)
                                   & (holiday_data['time'] < 20161101)]['time'].values  # strformat:%Y%m%d 20150302
    holidaydateList = map(datetools.DateStr_3ToDatestr_1, holidaydateList)  # strformat:%Y/%m/%d 2015/03/02
    holidaydateList = map(datetools.Datestr1_removezeros, holidaydateList)  # strformat:%Y/%m/%d 2015/3/2
    for i,cate in enumerate(cate3_list):
        print cate
        countlist=[]
        holidaycountlist = []
        shopidlist=shop_data[shop_data['cate3_name']==cate]['shopid'].values
        shop_Num=len(shopidlist)
        # print shopidlist
        for datestr in Datelist:
            cur_list=groupingData[(groupingData['time']==datestr) &
                                  (groupingData['shopid'].isin(shopidlist))]['count'].values
            # print cur_list
            cur_sum = 0
            if len(cur_list)!=0:
                cur_sum=np.sum(cur_list)

            #当前日期是假期的话就将当前值加入到假期序列中
            if datestr in holidaydateList:
                holidaycountlist.append(cur_sum)

            countlist.append(cur_sum)
        plot_series=pd.Series(countlist,index=Datelist)
        plot_series_holiday = pd.Series(holidaycountlist, index=holidaydateList)
        ###################数据设置完毕，开始画图#####################################
        plot_delta=datetime.timedelta(days=20)
        dates = drange(startDate, endDate, plot_delta)
        fig = plt.figure(figsize=(15, 8))
        plt.xlabel('date')
        plt.ylabel('count')
        plt.title('level3_cate:'+cate+'\n商家数:'+str(shop_Num))
        plt.grid(True)
        view_ax = fig.add_subplot(1, 1, 1)
        view_ax.set_xticklabels(dates, rotation=45, size=10)
        view_ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        view_ax.plot_date(plot_series.index, plot_series.values, 'm-', marker='.', linewidth=0.5)
        view_ax.plot_date(plot_series_holiday.index, plot_series_holiday.values, color='r', marker='.')
        plt.savefig(picturepath+str(i)+'_trend.png')
        view_ax.clear()
        plt.close()
# cate1_list=shop_data['cate1_name'].unique()
# cate2_list=shop_data['cate2_name'].unique()
# cate3_list=shop_data['cate3_name'].unique()
#
# for cate in cate1_list:
#     print cate=='美食'
#     print cate
# print '############################################################'
# for cate in cate2_list:
#     print cate
# print '############################################################'
# for cate in cate3_list:
#     print cate

if __name__=='__main__':
    shop_path = 'data/shop_info.txt'
    holiday_path='data/holiday.csv'
    real_data = pd.read_csv('data/user_pay_afterGrouping.csv')
    shop_data = pd.read_csv(shop_path, names=['shopid', 'city_name', 'location_id', 'per_pay', 'score', 'comment_cnt',
                                              'shop_level', 'cate1_name', 'cate2_name', 'cate3_name'])
    holiday_data=pd.read_csv(holiday_path,names=['time','flag'])
    countByDate(real_data,holiday_data,'all_shop_count.png')
    countByDateAndCate3(real_data,holiday_data,shop_data,'cate3figure/')
    countByDateAndCate1(real_data,holiday_data,shop_data,'cate1figure/')
    countByDateAndCate2(real_data,holiday_data,shop_data,'cate2figure/')