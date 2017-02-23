# encoding=utf-8
import Parameter as param
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from  matplotlib.dates  import num2date,datestr2num,drange,DateFormatter
from pylab import mpl
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

'''
本文件用于实现绘制线下预测数据与真实数据的之间的趋势图
'''
def set_ch():
    mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题



def getPredictList(predictedDF,shop_id):
    '''

    :param predictedDF:
    :param shop_id:
    :return:
    '''
    return predictedDF.loc[shop_id-1,'day1':'day14'].values

def getGroundTruthList(groundTruthDF,shop_id):
    '''

    :param groundTruthDF:
    :param shop_id:
    :return:
    '''
    countseries=groundTruthDF[groundTruthDF['shopid']==shop_id]['count'].values
    length=len(countseries)
    # print type(countseries[length-14:length-1])
    return countseries[length-15:length-1]


def produceOneFigure(groundTruthDF,predictedDF,shop_id,title,copy_path):
    '''
     绘制预测和真实值两者的序列
    :param groundTruthDF: 现实数据的真实值数据框
    :param predictedDF:   预测出来的数据的数据集框
    :param shop_id:       商店id
    :param title:         figure的标题
    :param copy_path:     figure保存路径
    :return:
    '''

    fig=plt.figure(figsize=(15,8))
    view_ax = fig.add_subplot(1, 1, 1)
    plt.xlabel('date')
    plt.ylabel('view_counts')
    xticklis=range(1,15)
    truthList=getGroundTruthList(groundTruthDF,shop_id)
    predictedList=getPredictList(predictedDF,shop_id)

    plt.grid(True)
    plt.title(title)
    #真实值用紫色
    view_ax.plot(xticklis,truthList,'m--', marker='.',linewidth=0.5)
    #预测值用蓝绿色
    view_ax.plot(xticklis, predictedList, 'c--', marker='.', linewidth=0.5);
    plt.savefig(copy_path)
    view_ax.clear()
    plt.close()




def produceOneGapFigure(groundTruthDF,predictedDF,shop_id,title,copy_path):
    '''
     绘制差值序列
    :param groundTruthDF: 现实数据的真实值数据框
    :param predictedDF:   预测出来的数据的数据集框
    :param shop_id:       商店id
    :param title:         figure的标题
    :param copy_path:     figure保存路径
    :return:             
    '''

    fig = plt.figure(figsize=(15, 8))
    view_ax = fig.add_subplot(1, 1, 1)
    plt.xlabel('date')
    plt.ylabel('view_counts')
    xticklis=range(1,15)
    truthList=getGroundTruthList(groundTruthDF,shop_id)
    predictedList=getPredictList(predictedDF,shop_id)
    predictedList=np.array(map(float,predictedList))
    truthList=np.array(map(float,truthList))
    gapList=abs(truthList-predictedList)
    plt.grid(True)
    plt.title(title)
    view_ax.plot(xticklis,gapList,'m--', marker='.',linewidth=0.5)

    plt.savefig(copy_path)
    view_ax.clear()

def produceOneScoreFigure(groundTruthDF,predictedDF,shop_id,title,copy_path):
    '''
    绘制分数序列
    :param groundTruthDF: 现实数据的真实值数据框
    :param predictedDF:   预测出来的数据的数据集框
    :param shop_id:       商店id
    :param title:         figure的标题
    :param copy_path:     figure保存路径  ..../../../xx.png
    :return:
    '''

    fig = plt.figure(figsize=(15, 8))
    view_ax = fig.add_subplot(1, 1, 1)
    plt.xlabel('date')
    plt.ylabel('view_counts')
    xticklis=range(1,15)
    truthList=getGroundTruthList(groundTruthDF,shop_id)
    predictedList=getPredictList(predictedDF,shop_id)
    predictedList=np.array(map(float,predictedList))
    truthList=np.array(map(float,truthList))

    scoreList=abs((truthList-predictedList)/(truthList+predictedList))
    plt.grid(True)
    plt.title(title)

    yticks=np.array(range(1,21))*0.05
    plt.ylim(0, 1)
    plt.yticks(yticks)
    # view_ax.set_yticklabels(yticks, size=20)
    view_ax.plot(xticklis,scoreList,'m--', marker='.',linewidth=0.5)
    plt.savefig(copy_path)
    view_ax.clear()


def GenerateAllContrastFig(groundTruthDF,predictedDF,shopDF,savepath,figtype):
    '''

    :param groundTruthDF:
    :param predictedDF:
    :param shopDF:
    :param figtype: 0:绘制两者序列  1:gap=Tg-Tp(差值序列) 2:gap=(Tg-Tp)/(Tg+Tp)(分数序列)
     :return:
    '''
    import  os
    if os.path.exists(savepath)==False:
        os.mkdir(savepath)
    ID_list=predict_data['shop_id'].unique()
    fig = plt.figure(figsize=(15, 8))
    view_ax = fig.add_subplot(1, 1, 1)
    for id_item in ID_list:
        # if id_item<499:
        #     continue
        print 'id:',id_item
        _cur_shop_info = shop_info[shop_info['shop_id'] == id_item]
        title= '\nshopID:' + str(_cur_shop_info['shop_id'].values[0]) + 'city:' + str(_cur_shop_info.city_name.values[0]) + \
              'perpay:' + str(_cur_shop_info.per_pay.values[0]) + '\nscore:' + str(_cur_shop_info.score.values[0]) + ' conmment:' \
                + str(_cur_shop_info.comment_cnt.values[0]) + 'cate1_name:' + str(_cur_shop_info.cate1_name.values[0]) + '\ncate2_name:' \
               + str(_cur_shop_info.cate2_name.values[0]) + 'cate3_name:' + str(_cur_shop_info.cate3_name.values[0])
        # print 'title:',title
        catename=_cur_shop_info.cate1_name.values[0]

        if type(catename)==float:
            catename=='nan'
        # 将中文按utf-8格式解码
        catename=catename.decode('utf-8')
        flodername=savepath +catename +'/'
        if os.path.exists(flodername) == False:
            os.mkdir(flodername)
        figure_path=flodername+str(id_item)+'_contrastscore.png'
        if figtype==0:
           produceOneFigure(groundTruthDF,predictedDF,id_item,title,figure_path)
        if figtype == 1:
            produceOneGapFigure(groundTruthDF, predictedDF, id_item, title, figure_path)
        if figtype == 2:
            produceOneScoreFigure(groundTruthDF, predictedDF, id_item, title, figure_path)

if __name__=='__main__':
    set_ch()
    # GenerateAllContrastFig(train_data,predict_data,shop_info)

    namelist = ['shop_id', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7', 'day8',
                'day9', 'day10', 'day11', 'day12', 'day13', 'day14']
    predict_data = pd.read_csv('result/result_revise_f4_train.csv', names=namelist)
    train_data=pd.read_csv(param.payAfterGroupingAndRevision_path)

    shop_path='data/shop_info.txt'

    shop_info=pd.read_csv(shop_path,names=['shop_id','city_name','location_id','per_pay',
										   'score','comment_cnt','shop_level','cate1_name','cate2_name','cate3_name'])
    savepath='figures_contrast/'
    # 使用方式：仿照被注释了的调用
    GenerateAllContrastFig(train_data,predict_data,shop_info,savepath,0)
    # produceOneFigure(train_data,predict_data,3,'..','3_c.png')
    # produceOneGapFigure(train_data,predict_data,3,'..','3_G.png')
    # produceOneScoreFigure(train_data,predict_data,3,'..','3_S.png')