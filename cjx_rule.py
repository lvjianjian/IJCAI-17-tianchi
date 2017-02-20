# encoding=utf-8
import numpy as np
import Parameter as para
import json
# import lightgbm as lgb
import pandas as pd


def nearestMean(train_data,savepath):
    result = np.zeros ([2000, 15])
    for shopid in range(1,2001):
        print 'shopid:',shopid
        count_values=train_data[train_data['shopid']==shopid]['count'].values
        week_num=3
        predict = []
        for i in range(14):
            values_len=len(count_values)
            cur_sum=0
            # scope=[0.167,0.501,0.334]     # nearestmean_len3_weightmedianmore.csv
            # scope=[0.334,0.501,0.167]       #  nearestmean_len3_weightmedianmore_v2.csv
            # scope=[0.501,0.334,0.167]        #  nearestmean_len3_weightmedianmore_v3.csv
            scope=[0.501,0.167,0.334]        #  nearestmean_len3_weightmedianmore_v4.csv
            for j in range(0,week_num):
                 # cur_sum=cur_sum +(j+1)*0.167*count_values[values_len-(j+1)*7]*week_num
                 cur_sum=cur_sum +scope[j]*count_values[values_len-(j+1)*7]*week_num
            cur_value=cur_sum/week_num
            predict.append(cur_value)
            # print count_values
            count_values=np.insert(count_values,values_len,cur_value)
        del count_values
        print 'predict:', predict
        result[shopid-1] = np.insert(predict, 0, shopid)
    result = pd.DataFrame (result.astype (np.int))
    result = result.sort_values (by=0).values
    if (savepath is not None):
        np.savetxt (savepath, result, delimiter=",", fmt='%d')
    else:
        print result
    return result

if __name__ == '__main__':
    train_data=pd.read_csv(para.payAfterGroupingAndRevisionCompletion_path)
    nearestMean(train_data,'result/nearestmean_len3_weightmedianmore_v4.csv')

