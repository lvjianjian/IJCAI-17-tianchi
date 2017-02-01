# encoding=utf-8
import pandas as pd
import time
import matplotlib.pyplot as plt
from  matplotlib.dates  import datestr2num
from  matplotlib.dates  import num2date,datestr2num,drange,DateFormatter
import datetime
import function_collection
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

weather_path='data/ijcai17-weather_1.csv'
shop_path='data/shop_info.txt'
pay_path='data/user_pay_afterGrouping.csv'
extra_view_path='data/extra_user_view_afterGrouping.csv'
view_path='data/user_view_afterGrouping.csv'
figure_pay_path='Figure\\';
index = pd.DatetimeIndex(['7/1/2015', '31/10/2016'])


######################################################################################
shop_info=pd.read_csv(shop_path,names=['shopid','city_name','location_id','per_pay','score','comment_cnt','shop_level','cate1_name','cate2_name','cate3_name'])
weather_info=pd.read_csv(weather_path)
print weather_info[weather_info.area=='三明']
# pay_data=pd.read_csv(pay_path)
# extra_view_data=pd.read_csv(extra_view_path,parse_dates=True)
# view_data=pd.read_csv(view_path,parse_dates=True)
# all_view_data=extra_view_data.append(view_data,ignore_index=True)
# #####################################################################################
# startDate=datetime.datetime(2015,6,1)
# endDate=datetime.datetime(2017,1,1)
# delta=datetime.timedelta(days=20)
# dates = drange(startDate , endDate, delta)
# # print view_data
# # view_data.append(extra_view_data,ignore_index=True)
# _shop_id_series = pay_data['shopid'].unique();
# fig = plt.figure(figsize=(15,8))
# view_ax=fig.add_subplot(1,1,1)
#
# plt.xlabel('date')
# plt.ylabel('view_counts')
#
# plt.xticks(dates)
#
#
# for shop_id in _shop_id_series:
#
#    if shop_id<247:
#       continue
#    view_ax.set_xticklabels(dates, rotation=45, size=5)
#    _cur_shop_info = shop_info[shop_info.shopid == shop_id ]
#    if str(_cur_shop_info.cate1_name.values[0]) == '美食':
#       figure_pay_path = 'M1_Figure\\';
#    else:
#       figure_pay_path = 'CS1_Figure\\';
#    # print _cur_shop_info
#    plt.title('\nshopID:'+str(_cur_shop_info.shopid.values[0])+' city:'+str(_cur_shop_info.city_name.values[0])
#              +' perpay:'+str(_cur_shop_info.per_pay.values[0])+'\nscore:'+str(_cur_shop_info.score.values[0])+' conmment:'+
#              str(_cur_shop_info.comment_cnt.values[0])+' cate1_name:'+str(_cur_shop_info.cate1_name.values[0])+'\ncate2_name:'+str(_cur_shop_info.cate2_name.values[0])+'cate3_name'+str(_cur_shop_info.cate3_name.values[0]))
#    ###############################################################################################################
#
#    _cur_date_series = all_view_data[all_view_data.shopid == shop_id]['time'].tolist()
#    _cur_count_series = all_view_data[all_view_data.shopid == shop_id]['count'].tolist()
#    _cd_series = pd.Series(_cur_count_series, index=_cur_date_series)
#    if len(_cd_series.index)!=0:
#       view_index = pd.DatetimeIndex([_cd_series.index[0], _cd_series.index[len(_cd_series)-1]])
#    else:
#       view_index = pd.DatetimeIndex(['2016-7-1', '2016-10-31'])
#
#    _view_series = pd.Series([0, 0], index=view_index).resample('D', ).pad()
#    for time in _cd_series.index:
#       _view_series[time] = _cd_series[time]
#    # view_cur_date_series =map(num2date, map(datestr2num, all_view_data[all_view_data.shopid == shop_id]['time'].tolist()))
#    # view_cur_count_series = all_view_data[all_view_data.shopid == shop_id]['count'].tolist()
#    # print _view_series.index
#    # print map(datestr2num, _view_series.index)
#    view_cur_date_series = _view_series.index
#    view_cur_count_series =_view_series.values
# ######################################################################################################
#    _cur_date_series = pay_data[pay_data.shopid == shop_id]['time'].tolist()
#    _cur_count_series = pay_data[pay_data.shopid == shop_id]['count'].tolist()
#    _cd_series = pd.Series(_cur_count_series, index=_cur_date_series)
#    pay_index = pd.DatetimeIndex([_cd_series.index[0], _cd_series.index[len(_cd_series)-1]])
#    _pay_series = pd.Series([0, 0], index=pay_index).resample('D', ).pad()
#    for time in _cd_series.index:
#       _pay_series[time] = _cd_series[time]
#    pay_cur_date_series = _pay_series.index
#    pay_cur_count_series = _pay_series.values
#    ########################################################################################################################
#    figure_name =figure_pay_path + str(shop_id) + '_view_time.png'
#    view_ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d') )
#    view_ax.plot_date(view_cur_date_series,view_cur_count_series,'m-', marker='.',linewidth=1);
#    view_ax.plot_date(pay_cur_date_series, pay_cur_count_series, 'k-', marker='.',linewidth=1);
#    print figure_name
#    plt.savefig(figure_name)
#    view_ax.clear()



