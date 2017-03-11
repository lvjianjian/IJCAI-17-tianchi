#encoding=utf-8
import pandas  as pd

projectPath = "/home/zhongjianlv/IJCAI/"
payAfterGrouping_path = projectPath + "data/user_pay_afterGrouping.csv"
payAfterGroupingAndRevision_path = projectPath + "data/user_pay_afterGroupingAndRevision.csv"
payAfterGroupingAndTurncate_path = projectPath + "data/user_pay_afterGroupingAndTurncate.csv"
payAfterGroupingAndRevisionAndCompletion_path = projectPath + "data/user_pay_afterGroupingAndRevisionAndCompletion.csv"
payAfterGroupingAndRevision2AndTurncate_path = projectPath + "data/user_pay_afterGroupingAndRevision2AndTurncate.csv"
payAfterGroupingAndRevision2AndCompletion_path = projectPath + "data/user_pay_afterGroupingAndRevision2AndCompletion.csv"
meanfiltered = projectPath + "processing_files/meanfiltered.csv"
meanfilteredAfterCompletion = projectPath + "processing_files/meanfilteredAfterCompletion.csv"
holidayPath = projectPath + "data/holiday.csv"
nearestmean_len3_weightmedianmore_v4_csv = projectPath+"result/nearestmean_len3_weightmedianmore_v4.csv"
fs182good = projectPath + "result/fs182good.csv"
dateparser1 = lambda dates:pd.datetime.strftime(dates, '%Y-%m-%d')
shopinfopath = projectPath + "data/shop_info.txt"
weather_info_path = projectPath + "data/weather_info.csv"

ignore_cb_shopids = [23, 627, 749, 1269, 1875]
ignore_ms_shopids = [5, 125, 284, 381, 411, 416, 434, 437, 444, 459, 470, 474, 501, 521, 524, 530, 619, 632, 654, 659, 660, 683, 700, 721, 727, 735, 742, 752, 768, 810, 956, 1050, 1058, 1100, 1107, 1145, 1163, 1185, 1214, 1241, 1243, 1380, 1384, 1407, 1447, 1462, 1464, 1486, 1510, 1526, 1548, 1556, 1567, 1609, 1650, 1681, 1716, 1730, 1747, 1769, 1803, 1831, 1835, 1856, 1858, 1893, 1918, 1968]
cb = [23, 66, 533, 561, 618, 627, 749, 894, 1218, 1397, 1520, 1632, 1740, 1925, 1973] #adaboost > 0.15
ignore_all_shopids = ignore_cb_shopids + ignore_ms_shopids
shop_info = None
weather_info = None

def getShopInfo():
    global shop_info
    if shop_info is None:
         shop_info = pd.read_csv(shopinfopath, names=["shopid","cityname","locationid","perpay","score","comment","level","cate1","cate2","cate3"])
    return shop_info

def extractShopValueByCate(cate_level,cate_name):
    shop_data = getShopInfo()
    return shop_data[shop_data['cate' + str(cate_level)] == cate_name]['shopid'].unique()


def getWeather_info():
    global weather_info
    if weather_info is None:
        weather_info = pd.read_csv(weather_info_path)
    return weather_info


if __name__ == "__main__":
    # import numpy as np
    # list = np.unique(getWeather_info()["weather"].values)
    # for value in list:
    #     if pd.isnull(value):
    #         continue
    #     if "小雨" in value:
    #         print value
    print len(extractShopValueByCate(1, "超市便利店"))