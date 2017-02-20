# encoding=utf-8
import pandas as pd
import time
import  matplotlib as mplt
mplt.use('Agg')
import matplotlib.pyplot as plt

from  matplotlib.dates  import datestr2num
from  matplotlib.dates  import num2date,datestr2num,drange,DateFormatter
import datetime
from datetime import datetime as ddt
import numpy as np
import  Parameter as para
import  function_collection as fc
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from pylab import mpl


score_info=[]



with open('LGBMProcessingfile/offline_resultlist.txt') as scorefile:
    sentencelist=scorefile.readlines()
    for sentence in sentencelist:
        score=float(sentence.split(':')[1])
        score_info.insert(len(score_info),score)
    scorefile.close()



# datestr='2015-01-10'
# datelist=map(num2date,map(datestr2num,['2015-1-10','2015-1-12']))
'''
<summary>解决保存图片中文显示的问题</summary>
<parameter>NULL</parameter>
<return>NULL</return>
'''

def set_ch():
    mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
set_ch()

def numDatetoStr2(DT):
   '''
   :param DT: input datetime
   :return: converted string;fomat:'%Y-%m-%d'
              2015-03-02
   '''
   temp_date = str(num2date(DT)).split(' ')[0]
   words = temp_date.split('-')
   return words[0] + '-' + words[1] + '-' + words[2]

def getRealValuesSeries(real_data,shopid):
    '''
    :param real_data:真实数据
    :param shopid:对应商家id
    :return:series
    '''
    count_list=real_data[real_data['shopid']==shopid]['count'].values
    datestr_list =real_data[real_data['shopid']==shopid]['time'].values
    timeAndCountSeries=pd.Series(count_list,index=datestr_list)
    full_index = pd.DatetimeIndex ([timeAndCountSeries.index[0], timeAndCountSeries.index[len(timeAndCountSeries) - 1]])
    _full_series = pd.Series ([0, 0], index=full_index).resample ('D', ).pad ()
    for time in timeAndCountSeries.index:
        _full_series[time] = timeAndCountSeries[time]
    return _full_series

def getpredictDataSeries(predict_data,shopid):
     startDate = datetime.datetime (2016, 11, 1)
     endDate = datetime.datetime (2016, 11, 15)
     delta = datetime.timedelta (days=1)
     predictIndex = map (numDatetoStr2, drange(startDate, endDate, delta))
     # _series = pd.Series ([0, 0], index=full_index).resample ('D', ).pad ()
     predictValues=predict_data[predict_data['shopid']==shopid].values[0,1:16]
     # print predictValues
     # print len(predictValues)
     # predictValues=predictrow[1:len(predictrow)]
     # predictIndex=pd.DatetimeIndex(['2016-11-1','2016-11-15'])
     # print type(predictIndex)
     predictIndex=pd.DatetimeIndex(predictIndex)
     predictSeries=pd.Series(predictValues,index=predictIndex)
     # [realValues,realIndex]=getSeriesTimeAndCount(real_data,shopid)
     # realValues.append(predictSeries)
     # fullValue=realValues
     # full_index = pd.DatetimeIndex([realIndex[0], realIndex[len(realIndex) - 1]])
     # fullSeries=pd.Series (fullValue, index=full_index)
     return predictSeries

def Figure_oneShop(shopid,shop_data,real_data,pre_data,socore_info=np.zeros(2000),figsavepath=None):
    startDate = datetime.datetime (2015, 6, 1)
    endDate = datetime.datetime (2017, 1, 1)
    delta = datetime.timedelta (days=20)
    dates = drange (startDate, endDate, delta)
    shop_id_series = real_data['shopid'].unique ();
    fig = plt.figure (figsize=(15, 8))
    plt.xlabel ('date')
    plt.ylabel ('counts')
    view_ax = fig.add_subplot (1, 1, 1)
    view_ax.set_xticklabels (dates, rotation=45, size=5)
    view_ax.xaxis.set_major_formatter (DateFormatter ('%Y-%m-%d'))
    ############################################################################################################
    shopid = int (shopid)
    child_path = 'normal_figurel_nonmeanfilter_nearmean_2/'
    _cur_shop_info = shop_data[shop_data.shopid == shopid]
    if socore_info is None:
        plt.title (
            '\nshopID:' + str (_cur_shop_info.shopid.values[0]) + ' city:' + str (
                _cur_shop_info.city_name.values[0]) + \
            ' perpay:' + str (_cur_shop_info.per_pay.values[0]) + '\nscore:' + str (
                _cur_shop_info.score.values[0]) + ' conmment:' \
            + str (_cur_shop_info.comment_cnt.values[0]) + 'cate1_name:' + str (
                _cur_shop_info.cate1_name.values[0]) + '\ncate2_name:' \
            + str (_cur_shop_info.cate2_name.values[0]) + 'cate3_name:' + str (_cur_shop_info.cate3_name.values[0]))
    else:
        plt.title (
            '\nshopID:' + str (_cur_shop_info.shopid.values[0]) + ' city:' + str (
                _cur_shop_info.city_name.values[0]) + \
            ' perpay:' + str (_cur_shop_info.per_pay.values[0]) + '\nscore:' + str (
                _cur_shop_info.score.values[0]) + ' conmment:' \
            + str (_cur_shop_info.comment_cnt.values[0]) + 'cate1_name:' + str (
                _cur_shop_info.cate1_name.values[0]) + '\ncate2_name:' \
            + str (_cur_shop_info.cate2_name.values[0]) + 'cate3_name:' + str (
                _cur_shop_info.cate3_name.values[0]) +
            '\n testscore:' + str (socore_info[shopid - 1]))
    PreValuesSeries = getpredictDataSeries (pre_data, shopid)
    RealValuesSeries = getRealValuesSeries (real_data, shopid)
    # print type(RealValuesSeries)
    figure_name = figsavepath + child_path + str (shopid) + '_trend_.png'
    view_ax.plot_date (RealValuesSeries.index, RealValuesSeries.values, 'm-', marker='.', linewidth=0.5)
    view_ax.plot_date (PreValuesSeries.index, PreValuesSeries.values, 'c-', marker='.', linewidth=0.5)
    print figure_name
    plt.savefig (figsavepath)
    view_ax.clear ()

def Figure_all(shop_data,real_data,pre_data,socore_info=np.zeros(2000),figsavepath=None):
    startDate = datetime.datetime (2015, 6, 1)
    endDate = datetime.datetime (2017, 1, 1)
    delta = datetime.timedelta (days=20)
    dates = drange (startDate, endDate, delta)
    shop_id_series = real_data['shopid'].unique ();
    fig = plt.figure (figsize=(15, 8))
    plt.xlabel ('date')
    plt.ylabel ('counts')
    view_ax = fig.add_subplot (1, 1, 1)
    view_ax.set_xticklabels (dates, rotation=45, size=5)
    view_ax.xaxis.set_major_formatter (DateFormatter ('%Y-%m-%d'))
############################################################################################################
    for shopid in shop_id_series:
        shopid=int(shopid)
        child_path = 'normal_figurel_nonmeanfilter_nearmean_2/'
        _cur_shop_info = shop_data[shop_data.shopid == shopid]
        if socore_info is None:
              plt.title (
                '\nshopID:' + str (_cur_shop_info.shopid.values[0]) + ' city:' + str (_cur_shop_info.city_name.values[0]) + \
                ' perpay:' + str (_cur_shop_info.per_pay.values[0]) + '\nscore:' + str (
                    _cur_shop_info.score.values[0]) + ' conmment:' \
                + str (_cur_shop_info.comment_cnt.values[0]) + 'cate1_name:' + str (
                    _cur_shop_info.cate1_name.values[0]) + '\ncate2_name:' \
                + str (_cur_shop_info.cate2_name.values[0]) + 'cate3_name:' + str (_cur_shop_info.cate3_name.values[0]))
        else:
            plt.title (
                '\nshopID:' + str (_cur_shop_info.shopid.values[0]) + ' city:' + str (_cur_shop_info.city_name.values[0]) + \
                ' perpay:' + str (_cur_shop_info.per_pay.values[0]) + '\nscore:' + str (
                    _cur_shop_info.score.values[0]) + ' conmment:' \
                + str (_cur_shop_info.comment_cnt.values[0]) + 'cate1_name:' + str (_cur_shop_info.cate1_name.values[0]) + '\ncate2_name:' \
                + str (_cur_shop_info.cate2_name.values[0]) + 'cate3_name:' + str (_cur_shop_info.cate3_name.values[0]) +
                '\n testscore:' + str(socore_info[shopid - 1]))
            if socore_info[shopid-1]>0.1:
                child_path = 'unnormal_figurel_nonmeanfilter_nearmean_2/'
        PreValuesSeries=getpredictDataSeries(pre_data,shopid)
        RealValuesSeries=getRealValuesSeries(real_data,shopid)
        # print type(RealValuesSeries)
        figure_name = figsavepath +child_path+ str (shopid) + '_trend_.png'
        view_ax.plot_date (RealValuesSeries.index, RealValuesSeries.values, 'm-', marker='.', linewidth=0.5)
        view_ax.plot_date (PreValuesSeries.index, PreValuesSeries.values, 'c-', marker='.', linewidth=0.5)
        print figure_name
        plt.savefig (figure_name)
        view_ax.clear()

if __name__=='__main__':
    shop_path = 'data/shop_info.txt'
    real_data = pd.read_csv ('data/user_pay_afterGrouping.csv')
    shop_data = pd.read_csv (shop_path, names=['shopid', 'city_name', 'location_id', 'per_pay', 'score', 'comment_cnt',
                                               'shop_level', 'cate1_name', 'cate2_name', 'cate3_name'])

    predict_data=pd.read_csv('result/nearestmean_len3.csv',names=para.predict_clonames)
    # plt.xticks (dates)

    # pd.Series().append()
    # print score_info
    # Figure_all(shop_data,real_data,predict_data,score_info,'predictedtrendFigs/')
    Figure_oneShop(23,shop_data,real_data,predict_data,score_info,'predictedtrendFigs/23_predict.png')
    # print JointpredictData(predict_data,real_data,1)