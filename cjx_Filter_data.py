# encoding=utf-8
import  pandas as pd
import  numpy
import  Parameter as para
from datetime import  datetime as ddt
import  datetime as dt
'''
本文件用于给原有的修正数据进行滤波
'''
revision_data=pd.read_csv(para.payAfterGroupingAndRevisionAndCompletion_path)
revision_data=revision_data[['shopid','time','count']]
def RemoveDatezero(datestr):
    '''

    :param datestr:%Y/%m/%d
    :return: 去掉日期个位数的0开头
    '''
    words=datestr.split('/')
    if words[1][0]=='0':
        words[1]=words[1][1]
    if words[2][0]=='0':
        words[2]=words[2][1]
    return words[0]+'/'+words[1]+'/'+words[2]


def getMeanFilterValue(shop_id,center_date,dataframe):
    '''
    :param shop_id:商店编号
    :param center_date:当前滤波日期(type:ddt)
    :param dataframe:数据框
    :return:平均值
    '''
    week_delta=dt.timedelta(days=7)
    weekday=ddt.isoweekday(center_date)  # 获取当天星期数
    last_weekdate=center_date-week_delta # 获取上一周的日期
    next_weekdate=center_date+week_delta # 获取下一周的日期
    # 将日期转化为string
    center_datestr=ddt.strftime(center_date,'%Y-%m-%d')
    lastweek_datestr=ddt.strftime(last_weekdate,'%Y-%m-%d')
    nextweek_datestr=ddt.strftime(next_weekdate,'%Y-%m-%d')
    center_values=dataframe[(dataframe['time']==center_datestr) & (dataframe['shopid']==shop_id) ]['count'].values
    last_values=dataframe[(dataframe['time']==lastweek_datestr) & (dataframe['shopid']==shop_id) ]['count'].values
    next_values=dataframe[(dataframe['time']==nextweek_datestr) & (dataframe['shopid']==shop_id) ]['count'].values
    sum_count=0
    numbers=0
    if len(center_values)!=0:
        sum_count=sum_count+center_values[0]
        numbers=1+numbers
    if len(last_values)!=0:
        sum_count = sum_count + last_values[0]
        numbers = 1 + numbers
    if len(next_values)!=0:
        sum_count = next_values[0]+sum_count
        numbers=1+numbers
    # print 'sum_count',sum_count
    # print 'numbers', numbers
    if sum_count==0:
        return 0
    else:
        # print sum_count/numbers
        return sum_count/numbers
    # return sum_count if sum_count==0 else sum_count/numbers


def MeanFilter(dataframe,file_Path):
    '''

    :param dataframe:用于滤波的数据
    :return: 新的数据源
    '''
    # 获取商品编号序列
    shopID_Values=dataframe['shopid'].unique()
    copy_dataframe=dataframe.copy()
    # print 'copy_df',copy_dataframe.head()
    for shopIDItem in shopID_Values:
        print 'shopID',shopIDItem
        #一次取出当前shopId的数据框 避免多次取出（空间换时间）

        curID_dataframe=dataframe[dataframe['shopid']==shopIDItem]
        dateStr_Values=curID_dataframe.time.values

        for dateStrItem in dateStr_Values:
            # 转化为ddt（datetime.datetime）类型
            dateItem=ddt.strptime(dateStrItem,'%Y-%m-%d')
            cur_value=getMeanFilterValue(shopIDItem,dateItem,curID_dataframe)
            index_Value=curID_dataframe[curID_dataframe['time']==dateStrItem].index.values[0]
            copy_dataframe.at[index_Value,'count']=cur_value
    copy_dataframe.to_csv(file_Path,index=False)
    return copy_dataframe






if __name__=='__main__':
     MeanFilter(revision_data,'processing_files/meanfilteredAfterCompletion.csv')

