#-*- coding=gbk -*-
import pandas as pd
import time as tm
import matplotlib.pyplot as plt
from  matplotlib.dates  import datestr2num
from  matplotlib.dates  import num2date,datestr2num,drange,DateFormatter
import datetime
from pylab import mpl

'''
<summary>解决保存图片中文显示的问题</summary>
<parameter>NULL</parameter>
<return>NULL</return>
'''
def set_ch():
    mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

def IsSubStr(list,fatherStr):
   '''
   :param list:listType
   :param fatherStr:
   :return: true or false
   '''
   for childStr in list:
      if childStr in fatherStr:
         return True
def Decode(obstr):
   '''

   :param obstr: 目标中文字符串
   :return:解码后的中文字符串
   '''
   return obstr.decode('gbk')


def numDatetoStr1(DT):
   '''
   :param DT: input datetime
   :return: converted string;fomat:'%Y\%m\%d'
   2015/3/2 not 2015/03/02
   '''
   temp_date = str(num2date(DT)).split(' ')[0]
   words = temp_date.split('-')
   if words[1][0] == '0':
      temp_str = words[1][1]
      words[1] = temp_str
   if words[2][0] == '0':
      temp_str = words[2][1]
      words[2] = temp_str
   return words[0] + '/' + words[1] + '/' + words[2]


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
   return datetime.datetime.strptime(datestr, '%Y/%m/%d')


def DateToStr_1(date):
   '''

   :param date: datetime
   :return: datestr: %Y-%m-%d
   '''
   return datetime.datetime.strftime(date, '%Y-%m-%d')


def StrDate1ToStrDate(datestr):
   '''

   :param datestr: %Y/%m/%d
   :return: datestr:%Y-%m-%d
   '''
   return DateToStr_1(StrToDate_1(datestr))
##############################################################
def Draw_Figure(pay_data,all_view_data,shop_id):
   shop_path = 'data/shop_info.txt'
   '''读取shop_info data'''
   shop_info = pd.read_csv(shop_path,
                           names=['shopid', 'city_name', 'location_id', 'per_pay', 'score', 'comment_cnt', 'shop_level',
                                  'cate1_name', 'cate2_name', 'cate3_name'])

   startDate = datetime.datetime(2015, 6, 1)
   endDate = datetime.datetime(2017, 1, 1)
   delta = datetime.timedelta(days=20)
   dates = drange(startDate, endDate, delta) #获取date刻度List
   fig = plt.figure(figsize=(15, 8))
   view_ax = fig.add_subplot(1, 1, 1)
   view_ax.set_xticklabels(dates, rotation=45, size=5)
   _cur_shop_info = shop_info[shop_info.shopid == shop_id ]
   if str(_cur_shop_info.cate1_name.values[0]) == '美食':
      figure_pay_path = 'M1_Figure\\';
   else:
      figure_pay_path = 'CS1_Figure\\';
   plt.title('\nshopID:'+str(_cur_shop_info.shopid.values[0])+' city:'+str(_cur_shop_info.city_name.values[0])
             +' perpay:'+str(_cur_shop_info.per_pay.values[0])+'\nscore:'+str(_cur_shop_info.score.values[0])+' conmment:'+
             str(_cur_shop_info.comment_cnt.values[0])+' cate1_name:'+str(_cur_shop_info.cate1_name.values[0])+'\ncate2_name:'+
             str(_cur_shop_info.cate2_name.values[0])+'cate3_name'+str(_cur_shop_info.cate3_name.values[0]))
   ##################绘制view_time折线图#############################################################################################

   _cur_date_series = all_view_data[all_view_data.shopid == shop_id]['time'].tolist()
   _cur_count_series = all_view_data[all_view_data.shopid == shop_id]['count'].tolist()
   _cd_series = pd.Series(_cur_count_series, index=_cur_date_series)
   if len(_cd_series.index)!=0:
      view_index = pd.DatetimeIndex([_cd_series.index[0], _cd_series.index[len(_cd_series)-1]])
   else:
      view_index = pd.DatetimeIndex(['2016-7-1', '2016-10-31'])

   _view_series = pd.Series([0, 0], index=view_index).resample('D', ).pad()
   for time in _cd_series.index:
      _view_series[time] = _cd_series[time]
   view_cur_date_series = _view_series.index
   view_cur_count_series =_view_series.values
###########################绘制pay_time折线图###########################################################################
   _cur_date_series = pay_data[pay_data.shopid == shop_id]['time'].tolist()
   _cur_count_series = pay_data[pay_data.shopid == shop_id]['count'].tolist()
   _cd_series = pd.Series(_cur_count_series, index=_cur_date_series)
   pay_index = pd.DatetimeIndex([_cd_series.index[0], _cd_series.index[len(_cd_series)-1]])
   _pay_series = pd.Series([0, 0], index=pay_index).resample('D', ).pad()
   for time in _cd_series.index:
      _pay_series[time] = _cd_series[time]
   pay_cur_date_series = _pay_series.index
   pay_cur_count_series = _pay_series.values
   ########################################################################################################################
   figure_name =figure_pay_path + str(shop_id) + '_view_time.png'
   view_ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d') )
   view_ax.plot_date(view_cur_date_series,view_cur_count_series,'m-', marker='.',linewidth=1);
   view_ax.plot_date(pay_cur_date_series, pay_cur_count_series, 'k-', marker='.',linewidth=1);
   print figure_name
   plt.savefig(figure_name)
   view_ax.clear()

##############################################################

def preprocess_Weather(weather_path):

   '''
   :param weather_path:the path of raw weather data
   :return: null
   yield the new weather data with weather level

   '''

   weather_info = pd.read_csv(weather_path, encoding="gb2312")
   weather_info['weather_level'] = 0
   level_0 = ''
   level_2 = map(Decode, ['雨', '雪'])
   level_1 = map(Decode, ['小雨'])
   for i, row in enumerate(weather_info.weather):
      row_date=weather_info.loc[i].date
      # print row_date
      _row_date=datetime.datetime.strptime(row_date,'%Y/%m/%d')
      _row_week=datetime.datetime.isoweekday(_row_date)
      if (_row_week >= 6):                       # 周末天气正常天气等级设置为3
            weather_info.loc[i, 'weather_level'] = 3

      if IsSubStr(level_2, str(row))  and (_row_week >= 6): # 周末天气雨雪等级设置为5
         weather_info.loc[i, 'weather_level'] = 5

      if IsSubStr(level_1, str(row))  and (_row_week >=6):    # 周末天气小雨等级设置为4
         weather_info.loc[i, 'weather_level'] = 4

      if IsSubStr(level_2, str(row)) and (_row_week<6):     # 非周末天气雨雪等级设置为2
         weather_info.loc[i, 'weather_level'] = 2

      if IsSubStr(level_1, str(row))  and (_row_week < 6):   # 非周末天气小雨等级设置为1
         weather_info.loc[i, 'weather_level'] = 1


   weather_info.to_csv('data\\weather_info.csv', encoding='gb2312', index=False)
# preprocess_Weather('data\\ijcai17-weather_1.csv')
def DatetoStr(DT):
   '''

   :param DT: number formatter datetime
   :return: converted string;fomat:'%Y\%m\%d'
   '''
   temp_date = str(num2date(DT)).split(' ')[0]
   words = temp_date.split('-')
   if words[1][0] == '0':
      temp_str = words[1][1]
      words[1] = temp_str
   if words[2][0] == '0':
      temp_str = words[2][1]
      words[2] = temp_str
   return words[0] + '/' + words[1] + '/' + words[2]

def Draw_Figure_unnormal(pay_data,all_view_data,weather_info,shop_id):
   '''
   绘制带有不正常天气标记的商家信息的折线图
   :param pay_data:
   :param all_view_data:
   :param weather_info:
   :param shop_id:
   :return:
   '''
   ###############################读取商户的基本信息################################################################################
   shop_path = 'data/shop_info.txt'
   '''读取shop_info data'''
   shop_info = pd.read_csv(shop_path,
                           names=['shopid', 'city_name', 'location_id', 'per_pay', 'score', 'comment_cnt', 'shop_level',
                                  'cate1_name', 'cate2_name', 'cate3_name'])

   ###############################设置时间的序列索引################################################################################
   startDate = datetime.datetime(2015, 6, 1)
   endDate = datetime.datetime(2017, 1, 1)
   delta = datetime.timedelta(days=20)
   dates = drange(startDate, endDate, delta)  # 用于设置画图刻度
   delt1 = datetime.timedelta(days=1)
   datelist = map(numDatetoStr1, drange(startDate, endDate, delt1))  # 用于获取天气信息时使用转化为str 格式:'%Y/%m/%d'

###############################画图############################################################################
   fig = plt.figure(figsize=(15, 8))
   view_ax = fig.add_subplot(1, 1, 1)
   view_ax.set_xticklabels(dates, rotation=45, size=5)

   #按shop_id获取shop_info信息
   _cur_shop_info = shop_info[shop_info.shopid == shop_id]

   #按经营类别分类
   if str(_cur_shop_info.cate1_name.values[0]) == '美食':
      figure_pay_path = 'food_figure\\';
   else:
      figure_pay_path = 'market_figure\\';
   #设置figure的title
   plt.title('\nshopID:' + str(_cur_shop_info.shopid.values[0]) + ' city:' + str(_cur_shop_info.city_name.values[0])
             + ' perpay:' + str(_cur_shop_info.per_pay.values[0]) + '\nscore:' + str(
      _cur_shop_info.score.values[0]) + ' conmment:' +
             str(_cur_shop_info.comment_cnt.values[0]) + ' cate1_name:' + str(
      _cur_shop_info.cate1_name.values[0]) + '\ncate2_name:' +
             str(_cur_shop_info.cate2_name.values[0]) + 'cate3_name' + str(_cur_shop_info.cate3_name.values[0]))
   #########################################################################################################################
   #获取shop_id对应的城市名
   city = _cur_shop_info.city_name.values[0]

   #获取对应城市的在时间段内天气信息
   object_weather = weather_info[(weather_info['area'] == city) & (weather_info['date'].isin(datelist))]
   print city, object_weather.area.values[0]
   #获取雨雪天气的时间序列（1，4为小雨）
   datelist_unnormal_1 = object_weather[
      (object_weather['weather_level'] == 1) | (object_weather['weather_level'] == 4)].date.values
   #获取雨雪天气的时间序列（2，5为雨雪）
   datelist_unnormal_2 = object_weather[
      (object_weather['weather_level'] == 2) | (object_weather['weather_level'] == 5)].date.values
   ###########################绘制view折线图####################################################################################

   _cur_date_series = all_view_data[all_view_data.shopid == shop_id]['time'].tolist()
   _cur_count_series = all_view_data[all_view_data.shopid == shop_id]['count'].tolist()
   _cd_series = pd.Series(_cur_count_series, index=_cur_date_series)
   #时间序列不为空，则使用时间序列的最大最小值形成的时间段进行采样
   if len(_cd_series.index) != 0:
      view_index = pd.DatetimeIndex([_cd_series.index[0], _cd_series.index[len(_cd_series) - 1]])
   else:
      #时间序列为空，则使用默认时间序列的最大最小值形成的时间段进行采样
      view_index = pd.DatetimeIndex(['2016-7-1', '2016-10-31'])
   #根据生产的view_index进行重采样(没有记录部分设为0)
   _view_series = pd.Series([0, 0], index=view_index).resample('D', ).pad()
   for time in _cd_series.index:
      _view_series[time] = _cd_series[time]
   view_cur_date_series = _view_series.index
   view_cur_count_series = _view_series.values
   ###########################绘制pay折线图####################################################################################

   _cur_date_series = pay_data[pay_data.shopid == shop_id]['time'].tolist()
   _cur_count_series = pay_data[pay_data.shopid == shop_id]['count'].tolist()
   #重采样
   _cd_series = pd.Series(_cur_count_series, index=_cur_date_series)
   pay_index = pd.DatetimeIndex([_cd_series.index[0], _cd_series.index[len(_cd_series) - 1]])
   _pay_series = pd.Series([0, 0], index=pay_index).resample('D', ).pad()
   for time in _cd_series.index:
      _pay_series[time] = _cd_series[time]

   pay_cur_date_series = _pay_series.index
   pay_cur_count_series = _pay_series.values
   pay_count_unnormal_1 = _pay_series[_pay_series.index.isin(
      map(StrDate1ToStrDate, datelist_unnormal_1))].values
   pay_date_unnormal_1 = _pay_series[_pay_series.index.isin(
      map(StrDate1ToStrDate, datelist_unnormal_1))].index
   pay_count_unnormal_2 = _pay_series[_pay_series.index.isin(
      map(StrDate1ToStrDate, datelist_unnormal_2))].values
   pay_date_unnormal_2 = _pay_series[_pay_series.index.isin(
      map(StrDate1ToStrDate, datelist_unnormal_2))].index
   ########################################################################################################################
   figure_name = figure_pay_path + str(shop_id) + '_view_time.png'
   view_ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
   # 绘制view_time折线图（紫色虚线）
   view_ax.plot_date(view_cur_date_series, view_cur_count_series, 'm--', marker='.', linewidth=0.5);
   # 绘制pay_time折线图（蓝绿色虚线）
   view_ax.plot_date(pay_cur_date_series, pay_cur_count_series, 'c--', marker='.', linewidth=0.5);
   #标记小雨天气的点（红色钻石状）
   view_ax.plot_date(pay_date_unnormal_1, pay_count_unnormal_1, color='r', marker='p')
   # 标记小雨天气的点（黄色钻石状）
   view_ax.plot_date(pay_date_unnormal_2, pay_count_unnormal_2, color='y', marker='p')
   print figure_name
   plt.savefig(figure_name)
   view_ax.clear()
   del view_ax
   del fig

def Draw_Figure_weekend(pay_data, all_view_data, weather_info, shop_id):
      '''
      绘制带有周末信息标记的商家信息的折线图
      :param pay_data:
      :param all_view_data:
      :param weather_info:
      :param shop_id:
      :return:
      '''
      ###############################读取商户的基本信息################################################################################
      shop_path = 'data/shop_info.txt'
      '''读取shop_info data'''
      shop_info = pd.read_csv(shop_path,
                              names=['shopid', 'city_name', 'location_id', 'per_pay', 'score', 'comment_cnt',
                                     'shop_level',
                                     'cate1_name', 'cate2_name', 'cate3_name'])

      ###############################设置时间的序列索引################################################################################
      startDate = datetime.datetime(2015, 6, 1)
      endDate = datetime.datetime(2017, 1, 1)
      delta = datetime.timedelta(days=20)
      dates = drange(startDate, endDate, delta)  # 用于设置画图刻度
      delt1 = datetime.timedelta(days=1)
      datelist = map(numDatetoStr1, drange(startDate, endDate, delt1))  # 用于获取天气信息时使用转化为str 格式:'%Y/%m/%d'

      ###############################画图############################################################################
      fig = plt.figure(figsize=(15, 8))
      view_ax = fig.add_subplot(1, 1, 1)
      view_ax.set_xticklabels(dates, rotation=45, size=5)

      # 按shop_id获取shop_info信息
      _cur_shop_info = shop_info[shop_info.shopid == shop_id]

      # 按经营类别分类（保存路径）
      if str(_cur_shop_info.cate1_name.values[0]) == '美食':
         figure_pay_path = 'M1_figure\\';
      else:
         figure_pay_path = 'CS1_figure\\';
      # 设置figure的title
      plt.title('\nshopID:' + str(_cur_shop_info.shopid.values[0]) + ' city:' + str(_cur_shop_info.city_name.values[0])
                + ' perpay:' + str(_cur_shop_info.per_pay.values[0]) + '\nscore:' + str(
         _cur_shop_info.score.values[0]) + ' conmment:' +
                str(_cur_shop_info.comment_cnt.values[0]) + ' cate1_name:' + str(
         _cur_shop_info.cate1_name.values[0]) + '\ncate2_name:' +
                str(_cur_shop_info.cate2_name.values[0]) + 'cate3_name' + str(_cur_shop_info.cate3_name.values[0]))

      #########################################################################################################################
      # 获取shop_id对应的城市名
      city = _cur_shop_info.city_name.values[0]

      # 获取对应城市的在时间段内天气信息
      object_weather = weather_info[(weather_info['area'] == city) & (weather_info['date'].isin(datelist))]
      print city, object_weather.area.values[0]
      # 获取周末的的时间序列（大于等于3的都是双休日）
      datelist_weekend = object_weather[
         object_weather['weather_level']>=3].date.values
      ###########################绘制view折线图####################################################################################

      _cur_date_series = all_view_data[all_view_data.shopid == shop_id]['time'].tolist()
      _cur_count_series = all_view_data[all_view_data.shopid == shop_id]['count'].tolist()
      _cd_series = pd.Series(_cur_count_series, index=_cur_date_series)
      # 时间序列不为空，则使用时间序列的最大最小值形成的时间段进行采样
      if len(_cd_series.index) != 0:
         view_index = pd.DatetimeIndex([_cd_series.index[0], _cd_series.index[len(_cd_series) - 1]])
      else:
         # 时间序列为空，则使用默认时间序列的最大最小值形成的时间段进行采样
         view_index = pd.DatetimeIndex(['2016-7-1', '2016-10-31'])
      # 根据生产的view_index进行重采样(没有记录部分设为0)
      _view_series = pd.Series([0, 0], index=view_index).resample('D', ).pad()
      for time in _cd_series.index:
         _view_series[time] = _cd_series[time]
      view_cur_date_series = _view_series.index
      view_cur_count_series = _view_series.values
      ###########################绘制pay折线图####################################################################################

      _cur_date_series = pay_data[pay_data.shopid == shop_id]['time'].tolist()
      _cur_count_series = pay_data[pay_data.shopid == shop_id]['count'].tolist()
      # 重采样
      _cd_series = pd.Series(_cur_count_series, index=_cur_date_series)
      pay_index = pd.DatetimeIndex([_cd_series.index[0], _cd_series.index[len(_cd_series) - 1]])
      _pay_series = pd.Series([0, 0], index=pay_index).resample('D', ).pad()
      for time in _cd_series.index:
         _pay_series[time] = _cd_series[time]

      pay_cur_date_series = _pay_series.index
      pay_cur_count_series = _pay_series.values
      pay_count_weekend = _pay_series[_pay_series.index.isin(
         map(StrDate1ToStrDate, datelist_weekend))].values
      pay_date_weekend = _pay_series[_pay_series.index.isin(
         map(StrDate1ToStrDate, datelist_weekend))].index
      ########################################################################################################################
      figure_name = figure_pay_path + str(shop_id) + '_view_time.png'
      view_ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
      # 绘制view_time折线图（紫色虚线）
      view_ax.plot_date(view_cur_date_series, view_cur_count_series, 'm--', marker='.', linewidth=0.5);
      # 绘制pay_time折线图（蓝绿色虚线）
      view_ax.plot_date(pay_cur_date_series, pay_cur_count_series, 'c--', marker='.', linewidth=0.5);
      # 标记小雨天气的点（红色五角星状）
      view_ax.plot_date(pay_date_weekend, pay_count_weekend, color='r', marker='p')
      print figure_name
      plt.savefig(figure_name)
      # fig.show()
      # tm.sleep(10)
      view_ax.clear()
      del view_ax
      del fig
set_ch()
