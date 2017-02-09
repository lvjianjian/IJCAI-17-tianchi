#-*- coding=utf-8 -*-

import Parameter as param
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from  matplotlib.dates  import num2date,datestr2num,drange,DateFormatter

pay_data = pd.read_csv(param.payAfterGrouping_path)
pay_revised_data = pd.read_csv(param.payAfterGroupingAndRevision_path)


def getDataFromStartToEnd(pay_data, shop_id):
    _cur_date_series = pay_data[pay_data.shopid == shop_id]['time'].tolist()
    _cur_count_series = pay_data[pay_data.shopid == shop_id]['count'].tolist()
    _cd_series = pd.Series(_cur_count_series, index=_cur_date_series)
    pay_index = pd.DatetimeIndex([_cd_series.index[0], _cd_series.index[len(_cd_series)-1]])
    _pay_series = pd.Series([0, 0], index=pay_index).resample('D', ).pad()
    for time in _cd_series.index:
        _pay_series[time] = _cd_series[time]

    pay_cur_date_series = _pay_series.index
    pay_cur_count_series = _pay_series.values
    return [pay_cur_date_series,pay_cur_count_series]

def getFigure_DataAndRevisionData(shop_id,data_from,data_end):
    startDate=datetime.datetime(2015,6,1)
    endDate=datetime.datetime(2017,1,1)
    delta=datetime.timedelta(days=20)
    dates = drange(startDate, endDate, delta)
    fig = plt.figure(figsize=(15,8))
    view_ax=fig.add_subplot(1,1,1)
    plt.xlabel('date')
    plt.ylabel('view_counts')
    plt.xticks(dates)
    view_ax.set_xticklabels(dates, rotation=45, size=10)
    view_ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    dataOfShopid = getDataFromStartToEnd(pay_data, shop_id)
    dataReviseOfShopid = getDataFromStartToEnd(pay_revised_data, shop_id)
    view_ax.plot_date(dataOfShopid[0], dataOfShopid[1], 'c--', marker='.',linewidth=0.5);
    view_ax.plot_date(dataReviseOfShopid[0], dataReviseOfShopid[1], 'r--', marker='.',linewidth=0.5);
    return view_ax


if __name__ == "__main__":
    getFigure_DataAndRevisionData(3,"","")
    plt.show()