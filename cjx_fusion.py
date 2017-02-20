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

def FusionCSV(stdData,addData,testscorelist,savepath):
    stdValues=stdData.values
    addValues=addData.values
    num=0
    for i in range(2000):
        #  stdData模型中线下结果不好店家用addData的替代 （>0.1）
        if(testscorelist[i]>0.065):
            print 'shopid:', str (i + 1)
            num+=1
            stdValues[i]=addValues[i]
    stdValues = pd.DataFrame (stdValues.astype (np.int))
    print num
    stdValues = stdValues.sort_values (by=0).values
    if (savepath is not None):
        np.savetxt (savepath, stdValues, delimiter=",", fmt='%d')
    else:
        print stdValues
    # return result
    print stdValues
    return stdValues



if __name__=='__main__':
    stdData=pd.read_csv('result/predict_seq14_lgbm_1_3_255_nonmeanf.csv',names=para.predict_clonames)
    addData=pd.read_csv('result/processed_nonmean_predict_lstm_14f2.csv',names=para.predict_clonames)
    savepath = 'result/fusion_predict_lgbmandlstmnonmean_14f2065.csv'
    # result_data = pd.read_csv ('result/predict_lstm_14f2.csv', names=para.predict_clonames)
    FusionCSV (stdData, addData,score_info,savepath)