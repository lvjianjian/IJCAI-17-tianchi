#-*- coding=utf-8 -*-

import Parameter as param
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from  matplotlib.dates  import num2date,datestr2num,drange,DateFormatter
import Parameter
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


def showLoss(losss):
    """
    显示loss和val_loss
    :param losss:
    :return:
    """
    loss = losss["loss"]
    val_loss = losss["val_loss"]
    fig, ax = plt.subplots(1, 1)
    plt.plot(np.array(loss),label="loss")
    plt.plot(np.array(val_loss),label="val_loss")
    ax.legend(loc="best")
    plt.show()

def show14Values(path):
    result = np.zeros(14)
    result2 = np.zeros(14)
    train_predict = np.loadtxt(path, delimiter=",", dtype=int)
    origin = pd.read_csv(Parameter.payAfterGrouping_path)
    reals = np.ndarray(0)
    shopids = train_predict.take(0, axis=1).tolist()
    for shopid in shopids:
        part_data = origin[origin.shopid == shopid]
        last_14_real_y = None
        # 取出一部分做训练集
        last_14_real_y = part_data[len(part_data) - 14:]["count"].values
        reals = np.append(reals,last_14_real_y)
    reals = reals.reshape((len(shopids),14))
    for k in range(len(shopids)):
        id = shopids[k]
        predict = train_predict[k][1:15]
        real = reals[k]
        for l in range(14):
            result[l] += abs(predict[l] - real[l])
            result2[l] += (predict[l] - real[l])
    fig,axes = plt.subplots(1, 2)

    axes[0].bar(left=np.arange(1, 15, 1), height=result)
    axes[0].set_title("abs(p-r)")
    axes[1].bar(left=np.arange(1, 15, 1), height=result2)
    axes[1].set_title("p-r")
    plt.show()

if __name__ == "__main__":
    # getFigure_DataAndRevisionData(3,"","")
    # plt.show()
    # loss = \
    #     {'loss': [0.00041882314018073239, 0.00037867650215314534, 0.00037378662505913214, 0.0003713799702834009, 0.00037005666031793933, 0.00036918804561976155, 0.00036865772146405803, 0.00036813014965927275, 0.00036778934275477604, 0.00036772622949158553, 0.00036744763723928583, 0.00036727980223762727, 0.00036719631093933648, 0.00036712848754963704], 'val_loss': [0.00058164982293166286, 0.00056453073596865835, 0.00056163064217487613, 0.00055091051334965363, 0.00054854475233677273, 0.0005462085495359717, 0.00054502757456651103, 0.00054351278854683331, 0.00054593499417540116, 0.00054160076715674833, 0.00054053679800337452, 0.00054162334159866621, 0.000541055027796229, 0.00054259252726038699]}
    # showLoss(loss)

    # show14Values(Parameter.projectPath + "/result/CNN_rt_hps60Last_0s_21d_21f_1_美食_40_3_20_sigmoid_1347shops_augmented_addNoiseInResult_train_1time.csv")
    show14Values(Parameter.projectPath + "/result/ANN1_rt_hps70Last_7s_0d_7f_1_超市便利店_40_3_10_sigmoid_569shops_augmented_train.csv")
    # show14Values("/home/zhongjianlv/IJCAI/lzj/final_result/CNN_ms_train.csv")