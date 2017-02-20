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
import  sys
import  cjx_Lgbm_model as clm
def processresult(result,savepath):
     processed_result=result.values
     for i in range(processed_result.shape[0]):
         processed_result[i]=clm.removeNegetive(processed_result[i])
         processed_result[i]=clm.toInt(processed_result[i])
     processed_result = pd.DataFrame (processed_result.astype (np.int))
     processed_result = processed_result.sort_values (by=0).values
     if (savepath is not None):
         np.savetxt (savepath, processed_result, delimiter=",", fmt='%d')
     else:
         print processed_result
     # return result
     print processed_result
     return processed_result


if __name__=='__main__':
    predict_path = 'result/nonmean_predict_lstm_14f2.csv'
    savepath='result/processed_nonmean_predict_lstm_14f2.csv'
    predict_data = pd.read_csv(predict_path,names=para.predict_clonames)
    processresult(predict_data,savepath)