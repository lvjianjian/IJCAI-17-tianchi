# encoding=utf-8

# Naive LSTM to learn three-char time steps to one-char mapping
import numpy as np
import Parameter as para
import json
import lightgbm as lgb
import pandas as pd
# from lv import  toInt
# from  lv import  removeNegetive

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# encoding=utf-8

# Naive LSTM to learn three-char time steps to one-char mapping
cur_thread_num = 20
# 序列长度设为7
seq_length =14
dateparser1 = para.dateparse1
#此文件已经做过均值平滑且填补完整
train_data = pd.read_csv (para.meanfiltered, date_parser=dateparser1)


def toInt(x):
    """
    将ndarray中的数字四舍五入
    :param x:
    :return:
    """
    for i in range(len(x)):
        x[i] = int(round(x[i]))
    return x


def removeNegetive(x):
    """
    去除负数，用1代替
    :param x:
    :return:
    """
    for i in range(len(x)):
       if(x[i]<0):
           x[i] = 1
    return x

def getCoutList (traindata, shopId):
    '''
    获取单个商店的countlist的内容
    :param traindata:单个店家的训练数据
    :param shopId: 商店ID
    :return: 商店ID对应的count序列（最好保持为float类型）
    '''
    countList = map (float, traindata[traindata['shopid'] == shopId]['count'].values)
    return countList[:len(countList)-30]


def preprocessCoutList (seq_length, counList):
    '''
    :param seq_length: 时间轴辐射长度
    :param counList:
    :return:
    '''
    dataX = []
    dataY = []
    # 将一天前两周时间的值化为特征向量组
    # 当天的值作为样本结果
    for i in range (0, len (counList) - seq_length, 1):
        seq_in = counList[i:i + seq_length]
        seq_out = counList[i + seq_length]
        dataX.append (seq_in)
        dataY.append (seq_out)
        # print seq_in, '->', seq_out
    return [dataX, dataY] # [[[],[],[]...],[v1,v2,v3..]]


def extrackTestDataFromCountlist(countlist,seq_length,test_length):
    '''
    根据特征序列长度截断countlist再提取测试的特征和groundtruth
    :param countlist:
    :param seq_length:
    :param test_length:
    :return:
    '''
    list_len=len(countlist)
    countlist=countlist[list_len-seq_length-test_length:list_len]
    return preprocessCoutList(seq_length,countlist)

def extrackTrainDataFromCountlist(countlist,seq_length,test_length):
    '''
    根据特征序列长度截断countlist再提取训练的特征和groundtruth

    :param countlist:
    :param seq_length:
    :param test_length:
    :return:
    '''
    list_len=len(countlist)
    countlist=countlist[:list_len-test_length]
    return preprocessCoutList(seq_length,countlist)

def ConcatFeatureAndGroundTruth(train_data,real_data,seq_length,test_length,shopId_Values):
     real_count_list=[]
     test_feature_list=[]
     train_feature_list=[]
     train_label_list=[]
     for shopId in shopId_Values:
          cur_countlist=train_data[train_data['shopid']==shopId]['count'].values
          [test_features,test_labels]=extrackTestDataFromCountlist(cur_countlist,seq_length,test_length)
          test_feature_list.extend(test_features)
          real_count_list.extend(test_labels)
          [train_features,train_labels]=extrackTrainDataFromCountlist(cur_countlist,seq_length,test_length)
          train_feature_list.extend(train_features)
          train_label_list.extend(train_labels)
     return [train_feature_list,train_label_list,test_feature_list,real_count_list]

def separateTrainAndEval(features,labels,shop_length):
     '''
     为lightgbm分割训练集合，验证集
     :param features:
     :param labels:
     :return:
     '''
     sample_len=len(features)
     train_X =[]
     train_Y =[]
     eval_X  =[]
     eval_Y  =[]
     iterations_num=sample_len/shop_length
     for i in range(shop_length):
         for j in range(i*iterations_num,((i + 1)*iterations_num-iterations_num/6)):
             train_X.append(features[j])
             train_Y.append(labels[j])
         for j in range(((i + 1) * iterations_num-iterations_num/6),(i+1)*iterations_num):
              eval_X.append(features[j])
              eval_Y.append(labels[j])
     return [train_X,train_Y,eval_X,eval_Y]

def BuildFreedomgbmTrainModel(features,labels,shop_length):
    # train
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2'},
        'num_leaves': 2048,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 10,
        'verbose': 0
    }
    boost_round = 1000
    early_stop  =  500
    ############################################################################
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'metric': {'l2'},
    #     'num_leaves': 2048,
    #     'learning_rate': 0.01,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 10,
    #     'verbose': 0
    # }
    # boost_round = 1000
    # early_stop = 500
    ################################################################################
    [train_feature, train_label,eval_feature,eval_label]=separateTrainAndEval(features,
                                                                              labels,
                                                                              shop_length)
    lgb_train = lgb.Dataset (train_feature, train_label)
    # create dataset for lightgbm
    lgb_eval = lgb.Dataset (eval_feature, eval_label, reference=lgb_train)
    gbm = lgb.train (params,
                     lgb_train,
                     num_boost_round=boost_round,
                     valid_sets=lgb_eval,
                     early_stopping_rounds=early_stop)


    return gbm

def predictWithTrainModel(model,test_features):

    return model.predict (test_features, num_iteration=model.best_iteration)

def popfirstColAndpushlstCol(features,predict_values):
    features_len=len(features)
    predict_values_len=len(predict_values)
    if features_len!=predict_values_len:
        print 'error'
    for i in range(predict_values_len):
        # del features[i][0]
        # print type(features)
        # print features
        features[i].append(predict_values[i])
        features [i]= features[i][1:len (features[i])]
    return features

def extractInitialFeatureforOnlinePre(train_data,seq_len,shopid):
    countlist = train_data[train_data['shopid'] == shopid]['count'].tolist()
    initial_feature=countlist[len(countlist)-seq_len:len(countlist)]
    return  [initial_feature]

def predictShopInCateGroup(train_data,real_data,shop_data,seq_length,test_length,catename,catelevel,IsOffline=True):
    '''

    :param train_data:
    :param real_data:
    :param shop_data:
    :param seq_length:
    :param test_length:
    :param catename:
    :param catelevel:
    :param IsOffline:
    :return:
    '''

    catelevel_name='cate'+str(catelevel)+'_name'
    shopId_Values=shop_data[shop_data[catelevel_name]==catename]['shopid'].unique()

    if IsOffline==False:
       test_length=0
    [train_feature_list, train_label_list,
     test_feature_list, real_count_list]=ConcatFeatureAndGroundTruth(train_data,real_data,
                                                                   seq_length,test_length,
                                                                   shopId_Values)

    shop_len=len(shopId_Values)
    #训练出model
    model=BuildFreedomgbmTrainModel(train_feature_list,train_label_list,shop_len)


    if IsOffline:
        # 预测
        predict_values = predictWithTrainModel (model, test_feature_list)
        predict_values=removeNegetive(predict_values)
        predict_values=toInt(predict_values)

        num_sc=scoreoneshop (predict_values, real_count_list)
        print 'score:',num_sc
        scorefile_name='lgbmfreedompara_catelevel'+str(catelevel)+'seqlen'+str(seq_length)+'_testlen'+str(test_length)+'.txt'
        if type(catename)==float:
            catename='nan'
        splitwords=catename.split ('/')
        if len(splitwords)>1:
            catename=splitwords[0]
        # print 'cate_level:',catelevel_name
        # print 'cate_name',catename.decode('utf-8')
        with open('processing_files/'+scorefile_name,'a') as scorefile:
            scorefile.write(catename+':'+str(num_sc)+'\n')
        return [predict_values,num_sc]
    else:
        result_val_list=[]
        zero_num=0
        for i,shopid in  enumerate(shopId_Values):
            predict_values=[]
            feature=extractInitialFeatureforOnlinePre(train_data,seq_length,shopid)
            for pre_num in range(14):
               pre_value= predictWithTrainModel(model,feature)
               # print pre_value
               # print feature
               # feature=np.insert(feature,len(feature)-1,pre_value[0])
               # feature=feature[1:len(feature)]
               # print feature
               # print '#################################'
               feature=popfirstColAndpushlstCol(feature,pre_value)#生成新的feature
               predict_values.append(pre_value[0])
               del pre_value
            if max(predict_values)<=0:
                zero_num=zero_num+1
            predict_values = removeNegetive (predict_values)
            predict_values = toInt (predict_values)
            print 'predict:',predict_values
            predict_values.insert(0,shopid)
            result_val_list.append(predict_values)
        print 'Allzero_num:',zero_num
        return result_val_list

def predictShopInWholeGroup(train_data,real_data,shop_data,seq_length,test_length,IsOffline=True):
    '''

    :param train_data:
    :param real_data:
    :param shop_data:
    :param seq_length:
    :param test_length:
    :param catename:
    :param catelevel:
    :param IsOffline:
    :return:
    '''

    shopId_Values=shop_data['shopid'].unique()

    if IsOffline==False:
       test_length=0
    [train_feature_list, train_label_list,
     test_feature_list, real_count_list]=ConcatFeatureAndGroundTruth(train_data,real_data,
                                                                   seq_length,test_length,
                                                                   shopId_Values)

    shop_len=len(shopId_Values)
    #训练出model
    model=BuildFreedomgbmTrainModel(train_feature_list,train_label_list,shop_len)


    if IsOffline:
        # 预测
        predict_values = predictWithTrainModel (model, test_feature_list)
        predict_values=removeNegetive(predict_values)
        predict_values=toInt(predict_values)

        num_sc=scoreoneshop (predict_values, real_count_list)
        print 'score:',num_sc
        scorefile_name='lgbmfreedompara_allgroup'+'seqlen'+str(seq_length)+'_testlen'+str(test_length)+'.txt'

        # print 'cate_level:',catelevel_name
        # print 'cate_name',catename.decode('utf-8')
        with open('processing_files/'+scorefile_name,'a') as scorefile:
            scorefile.write('all_group:'+str(num_sc)+'\n')
        return [predict_values,num_sc]
    else:
        zero_num=0
        result_val_list=[]
        for i,shopid in  enumerate(shopId_Values):
            predict_values=[]
            feature=extractInitialFeatureforOnlinePre(train_data,seq_length,shopid)
            for pre_num in range(14):
               pre_value= predictWithTrainModel(model,feature)
               # print pre_value
               # print feature
               # feature=np.insert(feature,len(feature)-1,pre_value[0])
               # feature=feature[1:len(feature)]
               # print feature
               # print '#################################'
               feature=popfirstColAndpushlstCol(feature,pre_value)#生成新的feature
               predict_values.append(pre_value[0])
               del pre_value
            if max(predict_values)<=0:
                zero_num=zero_num+1
            predict_values = removeNegetive (predict_values)
            predict_values = toInt (predict_values)
            print 'predict:',predict_values
            predict_values.insert(0,shopid)
            result_val_list.append(predict_values)
        print 'Allzero_num:',zero_num
        return result_val_list


def predictShopByClass_seq(train_data,real_data,shop_data,seq_length,test_length,cate_level,result_savepath,IsOffline=True):

    '''
    用序列做预测的方式
    :param train_data:
    :param real_data:
    :param shop_data:
    :param seq_length:
    :param test_length:
    :param class_level:
    :param result_savepath:
    :param scorelist_file:
    :param IsOffline:
    :return:
    '''
    if cate_level==0:
        if IsOffline:
            [predict_values, numsc] = predictShopInWholeGroup(train_data, real_data, shop_data, seq_length, test_length,IsOffline)

            print 'score:',numsc
        else:
            total_result = predictShopInWholeGroup(train_data, real_data, shop_data, seq_length, test_length,
                                        IsOffline)

            total_result = np.array (total_result)
            total_result = pd.DataFrame (total_result.astype (np.int))
            total_result = total_result.sort_values (by=0).values
            import os
            if os.path.exists (result_savepath) == False:
                os.mkdir (result_savepath)
            resultfile_name = 'processed_predict_lgbmfreedompara_wholeGroup'+ str (cate_level) + 'seqlen' + str (
                seq_length) + '.csv'
            print 'result_file:', resultfile_name
            np.savetxt (result_savepath + resultfile_name, total_result, delimiter=",", fmt='%d')

    else:
        catekey_name='cate'+str(cate_level)+'_name'
        cate_name_list = shop_info[catekey_name].unique()
        scorelist=[]
        i=0
        if IsOffline:
            for catename in cate_name_list:
                print catename
                i+=1
                print i

                [predict_values,numsc]=predictShopInCateGroup (train_data, real_data, shop_data, seq_length, test_length, catename, cate_level)
                scorelist.append(numsc)
            print 'total_score:',np.mean(scorelist)
        else:
            print 'predict...'
            print 'offline:',IsOffline
            total_result=[]
            for catename in cate_name_list:
                print catename
                i += 1
                print i
                result =  predictShopInCateGroup (train_data, real_data, shop_data, seq_length, test_length, catename, cate_level,IsOffline)
                total_result.extend(result)
            total_result=np.array(total_result)
            total_result = pd.DataFrame (total_result.astype (np.int))
            total_result = total_result.sort_values (by=0).values
            import os
            if os.path.exists(result_savepath)==False:
                os.mkdir(result_savepath)
            resultfile_name = 'processed_predict_lgbmfreedompara_catelevel' + str (cate_level) + 'seqlen' + str (
                seq_length) + '_testlen' + str (test_length) + '.csv'
            print 'result_file:',resultfile_name
            np.savetxt(result_savepath+resultfile_name,total_result,delimiter=",", fmt='%d')

def predictInTrainOneShop_Lgbm (train_data, seq_length, id_item):
    countList = getCoutList (train_data, id_item)
    #这里的14是验证集合长度
    [train_feature, train_label] = preprocessCoutList (seq_length, countList[:len(countList)-14])
    feature_len=len(train_feature)
    print len(train_feature)
    train_len = feature_len * 5/6 # 0.087
   # train_len = feature_len * 2/3 # 0.090
    eval_len  = feature_len - train_len

    train_x=train_feature[:train_len]
    train_y=train_label[:train_len]
    eval_x = train_feature[train_len:]
    eval_y = train_label[train_len:]
    # 生成测试集合
    [test_feature, test_label] = preprocessCoutList (seq_length,
                                                     countList[len (countList) - 14-seq_length:len (countList)])
    # [_feature, test_label] = preprocessCoutList (seq_length,
    #                                                  countList[len (countList) - 2 * 14:len (countList)])
    # create dataset for lightgbm
    lgb_train = lgb.Dataset (train_feature, train_label)
    # create dataset for lightgbm
    lgb_eval = lgb.Dataset (test_feature, test_label, reference=lgb_train)


    # specify your configurations as a dict
    # params = {
    #             'task': 'train',
    #             'boosting_type':'bagging',
    #             'objective': 'regression',
    #             'metric': {'l2','acc'},
    #             'learning_rate' : 0.1,
    #             'num_leaves' : 1024,
    #             'num_threads':16,
    #             'feature_fraction': 0.9,
    #             'bagging_fraction': 0.8,
    #             'bagging_freq': 5
    #             # 'min_data_in_leaf':0,
    #             # 'min_sum_hessian_in_leaf':100
    #         }
    # # print params
    print('Start training...')
    # train
    gbm = lgb.LGBMRegressor (objective='regression',
                             boosting_type='gbdt',

                             num_leaves=255,
                             # num_trees=500,
                             learning_rate=0.1,
                             n_estimators=60)
    # gbm = lgb.train (params,
    #
    #                  lgb_train,
    #
    #                  num_boost_round=20,
    #
    #                  valid_sets=lgb_eval,
    #
    #                  early_stopping_rounds=5)
    gbm.fit (train_x, train_y,
             eval_set=[(eval_x, eval_y)],
             eval_metric='l2',
             early_stopping_rounds=15)
    # estimator = lgb.LGBMRegressor (num_leaves=31)
    #
    # param_grid = {
    #
    #     'learning_rate': [0.01, 0.1, 1],
    #
    #     'n_estimators': [20, 40]
    #
    # }
    #
    # gbm = GridSearchCV (estimator, param_grid)

    # gbm.fit (train_feature, train_label)
    # print('Best parameters found by grid search are:', gbm.best_params_)
    # print('Save model...')
    # save model to file
    # gbm.save_model ('model/' + str (id_item) + 'lightgbm_model.txt')

    print('Start predicting...')
    # predict
    y_pred = gbm.predict (test_feature, num_iteration=gbm.best_iteration)
    # 返回两个一行的向量(1,14)
    print '#########################'+str(id_item)+'#############################'
    # print 'features:',test_feature
    # print 'pred:',y_pred
    # print 'real:',test_label
    curscore=scoreoneshop (y_pred, test_label)
    print 'score:',curscore
    with open('LGBMProcessingfile/offline_resultlistseq14_v1.txt','a') as rlistfile:
        rlistfile.write(str(id_item)+':'+str(curscore)+'\n')
    return [y_pred, test_label]


def predictInTrainOneShopOnline_Lgbm (train_data, seq_length, id_item):
    countList = getCoutList (train_data, id_item)
    [train_feature, train_label] = preprocessCoutList (seq_length, countList)
    feature_len = len (train_feature)
    print 'feature_len:',len (train_feature)
    train_len = feature_len *5 / 6
    # train_len = feature_len * 2 / 3        #  训练集长度（这里可以更换训练集提取策略来提升模型性能）
    eval_len = feature_len - train_len     #  验证集合长度
    train_x = train_feature[:train_len]
    train_y = train_label[:train_len]
    eval_x = train_feature[train_len:]
    eval_y = train_label[train_len:]
    # 生成测试集合
    # [test_feature, test_label] = preprocessCoutList (seq_length,
    #                                                  countList[len (countList) - 2 * 14:len (countList)])
    # [_feature, test_label] = preprocessCoutList (seq_length,
    #                                              countList[len (countList) - 2 * 14:len (countList)])
    # create dataset for lightgbm
    # lgb_train = lgb.Dataset (train_feature, train_label)
    # create dataset for lightgbm
    # lgb_eval = lgb.Dataset (test_feature, test_label, reference=lgb_train)

    # specify your configurations as a dict
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'metric': {'l2','acc'},
    #
    #     'learning_rate' : 0.1,
    #     'num_leaves' :255,
    #     'num_trees ':500,
    #     'num_threads' :16,
    #     'min_data_in_leaf ':0,
    #     'min_sum_hessian_in_leaf' :100
    # }
    # print params
    print('Start training...')
    # train
    gbm = lgb.LGBMRegressor (objective='regression',
                             num_leaves=255,
                             # num_trees=500,
                             learning_rate=0.1,
                             n_estimators=50)

    gbm.fit (train_x, train_y,
             eval_set=[(eval_x, eval_y)],
             eval_metric='l2',
             early_stopping_rounds=10)
    # estimator = lgb.LGBMRegressor (num_leaves=31)
    #
    # param_grid = {
    #
    #     'learning_rate': [0.01, 0.1, 1],
    #
    #     'n_estimators': [20, 40]
    #
    # }
    #
    # gbm = GridSearchCV (estimator, param_grid)

    # gbm.fit (train_feature, train_label)
    # print('Best parameters found by grid search are:', gbm.best_params_)
    # print('Save model...')
    # save model to file
    # gbm.save_model ('model/' + str (id_item) + 'lightgbm_model.txt')

    print('Start predicting...')
    y_pre_list=[]
    test_feature_pool = countList[len (countList)-seq_length :
                                  len(countList)]          # 先取后14天的
    for  predict_day in range(14):
        y_pred = gbm.predict ([test_feature_pool], num_iteration=gbm.best_iteration)
        test_feature_pool=np.insert(test_feature_pool,len(test_feature_pool),y_pred[0])
        y_pre_list=np.insert(y_pre_list,len(y_pre_list),y_pred[0])
        test_feature_pool = test_feature_pool[1:len (test_feature_pool)]    # 截断成特征向量
    # 返回两个一行的向量(1,14)
    print '#########################' + str (id_item) + '#############################'
    y_pre_list=toInt(y_pre_list)
    y_pre_list=removeNegetive(y_pre_list)
    print 'y_pre_list:',y_pre_list
    return y_pre_list

def predict_All_inTrain (train_data, seq_length, save_filename):
    result = np.zeros ((2000, 15))
    i = 0
    import os
    real = None
    predict = None
    shopid_values = train_data['shopid'].unique()
    print shopid_values
    for i, sid in enumerate (shopid_values):
        print sid
        # sid=int(sid)
        predictAndReal = predictInTrainOneShop_Lgbm (train_data, seq_length, sid)
        if real is None:
            real = predictAndReal[1]
        else:
            real = np.insert (real, len (real), predictAndReal[1])
        if predict is None:
            predict = predictAndReal[0]
        else:
            predict = np.insert (predict, len (predict), predictAndReal[0])
    result[i] = np.insert (predictAndReal[0], 0, sid)
    result = pd.DataFrame (result.astype (np.int))
    result = result.sort_values (by=0).values
    if (save_filename is not None):
        np.savetxt (save_filename, result, delimiter=",", fmt='%d')

    return [predict, real, result]

def predict_All_online (train_data, seq_length, save_filename):
    i = 0
    shopid_values = train_data['shopid'].unique()
    result = np.zeros ((2000, 15))
    print shopid_values
    for i in range (0,2000):
        print 'i:',i
        # sid=int(sid)
        predict= predictInTrainOneShopOnline_Lgbm(train_data, seq_length, shopid_values[i])
        result[i] = np.insert (predict, 0, shopid_values[i])
        # print result[i]
    result = pd.DataFrame (result.astype (np.int))
    result = result.sort_values (by=0).values
    if (save_filename is not None):
        np.savetxt (save_filename, result, delimiter=",", fmt='%d')
    else:
        print result
    return result

def scoreoneshop(predict, real):
    """
    评测公式
    :param predict: 预测值
    :param real: 真实值
    :return: 得分
    """
    # print "predict:", predict
    # print "real:", real
    score = 0
    length=len(predict)
    print length
    predict=np.array(predict)
    real=np.array(real)
    for i in range (length):
        score += (abs (predict[i] - real[i]) / (predict[i] + real[i]))
    score /= length
    return score


def score (predict, real):
    """
    评测公式
    :param predict: 预测值
    :param real: 真实值
    :return: 得分
    """
    # print "predict:", predict
    # print "real:", real
    score = 0
    for i in range (predict.shape[0]):
        score += (abs (predict[i] - real[i]) / (predict[i] + real[i]))
    score /= predict.shape[0]
    return score


if __name__ == '__main__':
    '''
    '''
    # print 'user_pay_afterGroupingAndRevisionAndCompletion.csv'
    # predictInTrainOneShopOnline_Lgbm(meanfiltered_data,7,1918)
    # prediceAndReal = predict_All_inTrain (train_data, seq_length, 'result/result_train_lgbm1_l2seq7v2.csv')
    # print score (prediceAndReal[0], prediceAndReal[1])
    cur_thread_num = 20;
    # 序列长度设为7
    seq_length =21
    test_length=14
    dateparser1 = para.dateparse1
    # 此文件已经做过均值平滑且填补完整
    shop_path = 'data/shop_info.txt'

    shop_info = pd.read_csv (shop_path, names=['shopid', 'city_name', 'location_id', 'per_pay',
                                               'score', 'comment_cnt', 'shop_level', 'cate1_name', 'cate2_name',
                                               'cate3_name'])
    # cate1_name_list=shop_info['cate1_name'].values
    # cate2_name_list=shop_info['cate2_name'].values
    # cate3_name_list=shop_info['cate3_name'].values
    train_data = pd.read_csv (para.payAfterGroupingAndRevision2Turncate_path)
    real_data=pd.read_csv(para.payAfterGroupingAndCompletionZeros)
    predictShopByClass_seq(train_data,real_data,shop_info,seq_length=seq_length,test_length=test_length,cate_level=0,result_savepath='result_v1/',IsOffline=False)
    # predictShopByClass_seq(train_data,real_data,shop_info,seq_length=seq_length,test_length=test_length,cate_level=2,result_savepath='',IsOffline=True)
    # predictShopInCateGroup(train_data,real_data,shop_info, seq_length=seq_length,test_length=test_length,
    #                        catename=cate1_name_list[0],catelevel=1)
    # print 'result/predict_seq141_lgbm_1_3_255_5f6_tu:',seq_length
    # predict_All_online(train_data,seq_length,'result/predict_seq141_lgbm_1_3_255_5f6_tu.csv') #此为2017/2/6日晚上所生成的预测文件（loss 为 mse）
    # predict_All_inTrain(train_data,seq_length,'result/predict_train_seq141_lgbm_1_3_255_nonmeanf.csv') #此为2017/2/6日晚上所生成的预测文件（loss 为 mse）
    # predictInTrainOneShop_Lgbm(train_data, seq_length, 3)
#