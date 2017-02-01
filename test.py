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
fc.set_ch()
shop_path='data/shop_info.txt'
weather_path='data/weather_info.csv'
pay_path='data/user_pay_afterGrouping.csv'
extra_view_path='data/extra_user_view_afterGrouping.csv'
view_path='data/user_view_afterGrouping.csv'
shop_info=pd.read_csv(shop_path,names=['shopid','city_name','location_id','per_pay','score','comment_cnt','shop_level','cate1_name','cate2_name','cate3_name'])
weather_info=pd.read_csv(weather_path,encoding='gb2312')
pay_data=pd.read_csv(pay_path)
extra_view_data=pd.read_csv(extra_view_path,parse_dates=True)
view_data=pd.read_csv(view_path,parse_dates=True)
all_view_data=extra_view_data.append(view_data,ignore_index=True)
fc.Draw_Figure_weekend(pay_data,all_view_data,weather_info,2)