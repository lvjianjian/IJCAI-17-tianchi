# encoding=utf-8
import pandas as pd
import time
import matplotlib.pyplot as plt
from  matplotlib.dates  import datestr2num
from  matplotlib.dates  import num2date,datestr2num,drange,DateFormatter
import datetime
import  function_collection as fc
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from pylab import mpl

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
shop_path='data/shop_info.txt'
weather_path='data/weather_info.csv'
pay_path='data/user_pay_afterGrouping.csv'
extra_view_path='data/extra_user_view_afterGrouping.csv'
view_path='data/user_view_afterGrouping.csv'
figure_pay_path='Figure\\';
index = pd.DatetimeIndex(['7/1/2015', '31/10/2016'])


######################################################################################
shop_info=pd.read_csv(shop_path,names=['shopid','city_name','location_id','per_pay','score','comment_cnt','shop_level','cate1_name','cate2_name','cate3_name'])
weather_info=pd.read_csv(weather_path,encoding='gb2312')
pay_data=pd.read_csv(pay_path)
extra_view_data=pd.read_csv(extra_view_path,parse_dates=True)
view_data=pd.read_csv(view_path,parse_dates=True)
all_view_data=extra_view_data.append(view_data,ignore_index=True)
#####################################################################################
startDate=datetime.datetime(2015,6,1)
endDate=datetime.datetime(2017,1,1)
delta=datetime.timedelta(days=20)
dates = drange(startDate , endDate, delta)
delt1=datetime.timedelta(days=1)
datelist= map(fc.numDatetoStr1,drange(startDate , endDate, delt1)) # datelist in the form of '%Y/%m/%d'

# datelis2= map(fc.numDatetoStr2,drange(startDate , endDate, delt1)) # datelist in the form of '%Y/%m/%d'
# print view_data
# view_data.append(extra_view_data,ignore_index=True)
_shop_id_series = pay_data['shopid'].unique();
fig = plt.figure(figsize=(15,8))
view_ax=fig.add_subplot(1,1,1)

plt.xlabel('date')
plt.ylabel('view_counts')

plt.xticks(dates)


for shop_id in _shop_id_series:

   # if shop_id<247:
   #    continue
   view_ax.set_xticklabels(dates, rotation=45, size=5)
   _cur_shop_info = shop_info[shop_info.shopid == shop_id ]
   # print '美食'
   if str(_cur_shop_info.cate1_name.values[0]) == '美食':
      figure_pay_path = 'food_figure\\';
   else:
      figure_pay_path = 'market_figure\\';
   # print 'Figure:',figure_pay_path
   # print 'cate_name:',str(_cur_shop_info.cate1_name.values[0])
   # print _cur_shop_info
   city=_cur_shop_info.city_name.values[0]
   # print 'city:',city
   # print 'perp:',str(_cur_shop_info.per_pay.values[0])
   # city=city.decode('gb2312')
   # _cur_title=
   # print _cur_title
   plt.title('\nshopID:'+str(_cur_shop_info.shopid.values[0])+' city:'+str(_cur_shop_info.city_name.values[0])+\
              ' perpay:'+str(_cur_shop_info.per_pay.values[0])+'\nscore:'+str(_cur_shop_info.score.values[0])+' conmment:'\
              +str(_cur_shop_info.comment_cnt.values[0])+'cate1_name:'+str(_cur_shop_info.cate1_name.values[0])+'\ncate2_name:'\
              +str(_cur_shop_info.cate2_name.values[0])+'cate3_name:'+str(_cur_shop_info.cate3_name.values[0]))
   object_weather=weather_info[(weather_info['area'] == city) & (weather_info['date'].isin(datelist))]
   print city,object_weather.area.values[0]
   # datelist_1 = object_weather[object_weather['weather_level']==1].date.values
   # datelist_2 = object_weather[object_weather['weather_level'] == 2].date.values
   # datelist_3 = object_weather[object_weather['weather_level'] == 3].date.values
   # datelist_4 = object_weather[object_weather['weather_level'] == 4].date.values
   datelist_unnormal_1 = object_weather[
      (object_weather['weather_level'] == 1) | (object_weather['weather_level'] == 4)].date.values
   datelist_unnormal_2 = object_weather[
      (object_weather['weather_level'] == 2) | (object_weather['weather_level'] == 5)].date.values
   # datelist_5 = object_weather[object_weather['weather_level'] == 5].date.values
   # datelist_weekend = object_weather[object_weather['weather_level'] >= 3].date.values
   ###############################################################################################################

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
######################################################################################################
   _cur_date_series = pay_data[pay_data.shopid == shop_id]['time'].tolist()
   _cur_count_series = pay_data[pay_data.shopid == shop_id]['count'].tolist()
   _cd_series = pd.Series(_cur_count_series, index=_cur_date_series)
   pay_index = pd.DatetimeIndex([_cd_series.index[0], _cd_series.index[len(_cd_series)-1]])
   _pay_series = pd.Series([0, 0], index=pay_index).resample('D', ).pad()
   for time in _cd_series.index:
      _pay_series[time] = _cd_series[time]

   pay_cur_date_series = _pay_series.index
   pay_cur_count_series = _pay_series.values
   pay_count_unnormal_1=_pay_series[_pay_series.index.isin(
      map(fc.StrDate1ToStrDate,datelist_unnormal_1))].values
   pay_date_unnormal_1=_pay_series[_pay_series.index.isin(
      map(fc.StrDate1ToStrDate,datelist_unnormal_1))].index
   pay_count_unnormal_2 = _pay_series[_pay_series.index.isin(
      map(fc.StrDate1ToStrDate, datelist_unnormal_2))].values
   pay_date_unnormal_2 = _pay_series[_pay_series.index.isin(
      map(fc.StrDate1ToStrDate, datelist_unnormal_2))].index
########################################################################################################################
   figure_name =figure_pay_path + str(shop_id) + '_view_time.png'
   view_ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d') )
   view_ax.plot_date(view_cur_date_series,view_cur_count_series,'m--', marker='.',linewidth=0.5);
   view_ax.plot_date(pay_cur_date_series, pay_cur_count_series, 'c--', marker='.',linewidth=0.5);
   view_ax.plot_date(pay_date_unnormal_1,pay_count_unnormal_1,color='r',marker='p')
   view_ax.plot_date(pay_date_unnormal_2, pay_count_unnormal_2, color='y', marker='p')
   print figure_name
   plt.savefig(figure_name)
   view_ax.clear()



