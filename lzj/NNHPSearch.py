# encoding=utf-8

from RNN.ANN import my_loss
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD,Adadelta,Adagrad,Adam,Adamax,Nadam
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from FeatureExtractor import *
from cjx_predict import scoreoneshop
from lv import removeNegetive,toInt
import numpy as np
from keras.layers import Dense,Lambda
from keras.layers import SimpleRNN
from keras.models import Sequential,Model
from keras.utils import np_utils
import Parameter
import pandas as pd
from cjx_predict import getCoutList
from cjx_predict import preprocessCoutList
from keras.optimizers import RMSprop
from keras.regularizers import l2, activity_l2,l1
from keras import backend as K
from keras.layers import BatchNormalization
from lv import toInt,removeNegetive
from sklearn.preprocessing import MinMaxScaler
import datetime
from cjx_predict import scoreoneshop, score
from keras.models import model_from_json

#指定种子以重现ANN结果
np.random.seed(128)


def saveModel(save_path,model):
    if isinstance(model,Model):
        json_string = model.to_json()  #等价于 json_string = model.get_config()
        open(save_path,'w').write(json_string)
        model.save_weights(save_path+".h5")

def getModel(save_path):
    model = model_from_json(open(save_path).read())
    model.load_weights(save_path+".h5")
    return model



adagrad = Adagrad(0.01)
loss_function='mse'

# Function to create model, required for KerasRegressor
def create_model(input_dim, h1_unit = 16, optimizer = adagrad, init = "normal",h1_activation="relu"):
    # create model
    model = Sequential()
    model.add(Dense(h1_unit, init=init, input_dim=input_dim, activation=h1_activation)) #sigmoid
    model.add(Dense(1, init=init, activation='linear', activity_regularizer=activity_l2(0.01)))
    # Compile model
    model.compile(loss=loss_function, optimizer=optimizer)
    return model

# Function to create model, required for KerasRegressor
def create_model2(input_dim, h1_unit = 16, h2_unit = 8,optimizer = "adagrad", init = "normal"):
    # create model
    model = Sequential()
    model.add(Dense(h1_unit, init=init, input_dim=input_dim, activation="relu")) #sigmoid
    model.add(Dense(h2_unit,init=init,activation="relu"))
    model.add(Dense(1, init=init, activation='linear', activity_regularizer=activity_l2(0.01)))
    # Compile model
    model.compile(loss='mse', optimizer=optimizer)
    return model



def predictAllShop_ANN1_HPS(all_data, trainAsTest=False, saveFilePath = None, featurePath = None,
                            cate_level = 0, cate_name = None, featureSavePath = None, needSaveFeature = False ,ignore_shopids = []
                            , needCV = False,model_path = None, Augmented = False,
                            ignore_get_train = True, ignore_predict = True):
    """
    通过gridsearch找超参数
    :param trainAsTest: 是否使用训练集后14天作为测试集
    :param saveFilePath
    :param featurePath:
    :param cate_level:
    :param cate_name:
    :param featureSavePath:
    :param needSaveFeature:
    :param ignore_shopids:
    :param create_model_function:
    :param needCV
    :param Augmented:是否增广样本
    :param  ignore_get_train:是否忽略获取样本
    :param ignore_predict:是否忽略预测
    :return:
    """

    verbose = 2
    last_N_days = 70
    #记录已经被忽略的商店数量
    # ignores = 0
    shop_need_to_predict = 2000
    if(cate_level is 0):
        shopids = np.arange(1, 1 + shop_need_to_predict, 1)
    else:
        shopids = Parameter.extractShopValueByCate(cate_level, cate_name)
    shop_info = pd.read_csv(Parameter.shopinfopath, names=["shopid", "cityname", "locationid", "perpay", "score", "comment", "level", "cate1", "cate2", "cate3"])

    weather = False
    weekOrWeekend = False
    day_back_num = 0
    sameday_backNum = 7
    week_backnum = 3
    other_features = [statistic_functon_mean, statistic_functon_median]
    other_features = []
    shop_features = ["perpay", "comment", "score", "level"]
    shop_features = []
    #是否是周末hot_encoder
    hot_encoder = onehot([[1], [0]])
    #类别1hot_encoder
    cate1_list = np.unique(shop_info['cate1'])
    cate1_label_encoder = labelEncoder(cate1_list)
    cate1_list2 = cate1_label_encoder.transform(cate1_list).reshape((-1, 1))
    cate1_hot_encoder = onehot(cate1_list2)


    if featurePath is None:
        all_x = None
        all_y = None
        for shopid in shopids:
            if ignore_get_train:
                if shopid in ignore_shopids:
                    print "ignore get train", shopid
                    continue
            print "get ", shopid, " train"
            part_data = all_data[all_data.shopid == shopid]
            last_14_real_y = None
            # 取出一部分做训练集
            if trainAsTest: #使用训练集后14天作为测试集的话，训练集为前面部分
                last_14_real_y = part_data[len(part_data) - 14:]["count"].values
                part_data = part_data[0:len(part_data) - 14]
            # print last_14_real_y
            '''确定跳过前面多少天的数据'''
            skipNum = part_data.shape[0] - last_N_days
            if skipNum < 0:
                skipNum = 0
            train_x = None
            '''获取特征'''
            if sameday_backNum != 0: #sameday
                sameday = extractBackSameday(part_data, sameday_backNum, skipNum, nan_method_sameday_mean)
                train_x = getOneWeekdayFomExtractedData(sameday)
            if day_back_num != 0: #day
                if train_x is not None:
                    train_x = np.concatenate((train_x, getOneWeekdayFomExtractedData(extractBackDay(part_data, day_back_num, skipNum, nan_method_sameday_mean))),axis=1)
                else:
                    train_x = getOneWeekdayFomExtractedData(extractBackDay(part_data, day_back_num, skipNum, nan_method_sameday_mean))
            if weekOrWeekend: #weekOrWeekend
                ws = getOneWeekdayFomExtractedData(extractWorkOrWeekend(part_data, skipNum))
                train_x = np.concatenate((train_x, hot_encoder.transform(ws)), axis=1)


            count = extractCount(part_data, skipNum)
            train_y = getOneWeekdayFomExtractedData(count)
            for feature in other_features:
                value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, skipNum, nan_method_sameday_mean, feature))
                train_x = np.append(train_x, value, axis=1)

            '''添加商家信息'''
            # print train_x,train_x.shape
            index = shopid - 1
            oneshopinfo = shop_info.ix[index]
            shop_city = oneshopinfo['cityname']
            shop_perpay = oneshopinfo['perpay'] if not pd.isnull(oneshopinfo['perpay']) else 0
            shop_score = oneshopinfo['score'] if not pd.isnull(oneshopinfo['score']) else 0
            shop_comment = oneshopinfo['comment'] if not pd.isnull(oneshopinfo['comment']) else 0
            shop_level = oneshopinfo['level'] if not pd.isnull(oneshopinfo['level']) else 0
            shop_cate1 = oneshopinfo['cate1']
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=DeprecationWarning)
                shop_cate1_encoder = cate1_hot_encoder.transform(cate1_label_encoder.transform([shop_cate1]))
            if "perpay" in shop_features:
                train_x = np.insert(train_x, train_x.shape[1], shop_perpay, axis=1)
            if "score" in shop_features:
                train_x = np.insert(train_x, train_x.shape[1], shop_score, axis=1)
            if "comment" in shop_features:
                train_x = np.insert(train_x, train_x.shape[1], shop_comment, axis=1)
            if "level" in shop_features:
                train_x = np.insert(train_x, train_x.shape[1], shop_level, axis=1)
            if "cate1" in shop_features:
                for i in range(shop_cate1_encoder.shape[1]):
                    train_x = np.insert(train_x, train_x.shape[1], shop_cate1_encoder[0][i], axis=1)
            '''商家信息添加完毕'''

            '''天气特征'''
            if weather:
                weathers = getOneWeekdayFomExtractedData(extractWeatherInfo(part_data, skipNum, shop_city))
                train_x = np.append(train_x, weathers, axis=1)
            '''天气特征结束'''

            if all_x is None:
                all_x = train_x
                all_y = train_y
            else:
                all_x = np.insert(all_x,all_x.shape[0],train_x,axis=0)
                all_y = np.insert(all_y,all_y.shape[0],train_y,axis=0)

                # '''添加周几'''
                # extract_weekday = getOneWeekdayFomExtractedData(extractWeekday(part_data, skipNum))
                # train_x = np.append(train_x, extract_weekday, axis=1)
                # ''''''

                # train_x = train_x.reshape((train_x.shape[0],
                #                            train_x.shape[1], 1))
                # print model.get_weights()
                # part_counts = []
                # for i in range(7):
                #     weekday = i + 1
                #     part_count = getOneWeekdayFomExtractedData(count, weekday)
                #     part_counts.append(part_count)


        train_x = all_x
        train_y = all_y

        if needSaveFeature:
            featureAndLabel = np.concatenate((train_x, train_y), axis=1)
            flDF = pd.DataFrame(featureAndLabel)
            if featureSavePath is None:
                if trainAsTest:
                    featureSavePath = Parameter.projectPath + "lzj/train_feature/%dCatelevel_%sCatename_%dfeatures_%dSameday_%dDay_%dLast.csv" % (cate_level,cate_name,flDF.shape[1] - 1,sameday_backNum,day_back_num,last_N_days)
                else:
                    featureSavePath = Parameter.projectPath + "lzj/feature/%dCatelevel_%sCatename_%dfeatures_%dSameday_%dDay_%dLast.csv" % (cate_level, cate_name, flDF.shape[1] - 1, sameday_backNum, day_back_num, last_N_days)
            flDF.to_csv(featureSavePath)
    else:#有featurePath文件
        if trainAsTest:
            path = Parameter.projectPath + "lzj/train_feature/"+featurePath
        else:
            path = Parameter.projectPath + "lzj/feature/" + featurePath
        flDF = pd.read_csv(path, index_col=0)
        train_x = flDF.values[:, :-1]
        train_y = flDF.values[:, -1:]
        # print train_x
        # print train_y




    """增广训练集"""
    if Augmented:
        def augument_relu(v):
            if v <= 50:
                return v
            if v > 50 and v <= 100:
                return v*1.05
            elif v > 100:
                return v*1.1

        end = train_x.shape[0]
        for index in range(end):
            train_x = np.concatenate((train_x, [train_x[index]]), axis=0)
            ov = train_y[index][0]
            train_y = np.concatenate((train_y, [[augument_relu(ov)]]), axis=0)

    '''将t标准化'''
    x_scaler = MinMaxScaler().fit(train_x)
    y_scaler = MinMaxScaler().fit(train_y)
    train_x = x_scaler.transform(train_x)
    train_y = y_scaler.transform(train_y)
    '''标准化结束'''


    if model_path is None:
        if needCV:
            '''gridsearchCV'''
            # nb_epoch=rnn_epoch, batch_size=batch_size, verbose=verbose
            # input_dim, h1_unit = 16, optimizer = "adagrad", init = "normal"):
            input_dim = [train_x.shape[1]]
            h1_activation = ["relu"]
            h1_unit = [8, 12, 16, 20]
            model = KerasRegressor(build_fn=create_model, verbose=verbose)
            batch_size = [3, 5, 7, 10]
            epochs = [10, 15, 20, 25, 30]
            param_grid = dict(batch_size = batch_size, nb_epoch = epochs, h1_unit = h1_unit, input_dim = input_dim)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,scoring="neg_mean_squared_error")
            grid.refit = False
            grid_result = grid.fit(train_x, train_y)

            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            for params, mean_score, scores in grid_result.grid_scores_:
                print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

        if not needCV:
            input_dim = train_x.shape[1]
            h1_unit = 10
            h1_activation = "sigmoid"
            batch_size = 3
            epochs = 40

        else:
            input_dim = train_x.shape[1]
            epochs = grid_result.best_params_['nb_epoch']
            batch_size = grid_result.best_params_['batch_size']
            h1_unit = grid_result.best_params_["h1_unit"]
            h1_activation = "sigmoid"

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        best_model = create_model(input_dim=input_dim, h1_unit=h1_unit, h1_activation = h1_activation)
        hist = best_model.fit(train_x, train_y, verbose=verbose, batch_size=batch_size, nb_epoch=epochs,  validation_split = 0.1, callbacks=[early_stopping])
        print hist.history


        #保存模型
        if trainAsTest:
            model_save_path = Parameter.projectPath+"lzj/train_model/" + \
                              "%dlast_%ds_%dd_%df_%d_%s_%d_%d_%d_%s.json" \
                              % (last_N_days,sameday_backNum, day_back_num, train_x.shape[1], cate_level, cate_name
                                 , epochs, batch_size, h1_unit, h1_activation)
            saveModel(model_save_path,best_model)
        else:
            model_save_path = Parameter.projectPath+"lzj/model/" + \
                              "%dlast_%ds_%dd_%df_%d_%s_%d_%d_%d_%s.json" \
                              % (last_N_days,sameday_backNum, day_back_num, train_x.shape[1], cate_level, cate_name
                                 ,  epochs, batch_size, h1_unit, h1_activation)
            saveModel(model_save_path,best_model)
    else:#model_path is not none
        print "get model from " + model_path
        best_model = getModel(model_path)

    format = "%Y-%m-%d"
    if trainAsTest:
        startTime = datetime.datetime.strptime("2016-10-18", format)
    else:
        startTime = datetime.datetime.strptime("2016-11-1", format)
    timedelta = datetime.timedelta(1)

    '''预测商家'''
    model = best_model
    preficts_all = None
    real_all = None

    for j in shopids:
        if ignore_predict:
            if j in ignore_shopids:
                print "ignore predict", j
                # ignores += 1
                continue
        print "predict:", j
        preficts = []
        part_data = all_data[all_data.shopid == j]
        last_14_real_y = None

        if trainAsTest: #使用训练集后14天作为测试集的话，训练集为前面部分
            last_14_real_y = part_data[len(part_data) - 14:]["count"].values
            part_data = part_data[0:len(part_data) - 14]

        '''预测14天'''
        for i in range(14):
            currentTime = startTime + timedelta * i
            strftime = currentTime.strftime(format)
            # index = getWeekday(strftime) - 1
            # part_count = part_counts[index]
            #取前{sameday_backNum}周同一天的值为特征进行预测
            part_data = part_data.append({"count":0,"shopid":j,"time":strftime,"weekday":getWeekday(strftime)},ignore_index=True)
            x = None
            if sameday_backNum != 0:
                x = getOneWeekdayFomExtractedData(extractBackSameday(part_data,sameday_backNum,part_data.shape[0] - 1, nan_method_sameday_mean))
            if day_back_num != 0:
                if x is None:
                    x = getOneWeekdayFomExtractedData(extractBackDay(part_data,day_back_num,part_data.shape[0] -1 ,nan_method_sameday_mean))
                else:
                    x = np.concatenate((x, getOneWeekdayFomExtractedData(extractBackDay(part_data,day_back_num,part_data.shape[0] -1 ,nan_method_sameday_mean))),axis=1)
            if weekOrWeekend:
                x = np.concatenate((x, hot_encoder.transform(getOneWeekdayFomExtractedData(extractWorkOrWeekend(part_data,part_data.shape[0] - 1)))),axis=1)

            for feature in other_features:
                x_value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, part_data.shape[0]-1, nan_method_sameday_mean, feature))
                x = np.append(x, x_value, axis=1)
            # '''添加周几'''
            # x = np.append(x, getOneWeekdayFomExtractedData(extractWeekday(part_data, part_data.shape[0]-1)), axis=1)
            # ''''''

            '''添加商家信息'''
            index = j - 1
            oneshopinfo = shop_info.ix[index]
            shop_city = oneshopinfo["cityname"]
            shop_perpay = oneshopinfo['perpay'] if not pd.isnull(oneshopinfo['perpay']) else 0
            shop_score = oneshopinfo['score'] if not pd.isnull(oneshopinfo['score']) else 0
            shop_comment = oneshopinfo['comment'] if not pd.isnull(oneshopinfo['comment']) else 0
            shop_level = oneshopinfo['level'] if not pd.isnull(oneshopinfo['level']) else 0
            if "perpay" in shop_features:
                x = np.insert(x, x.shape[1], shop_perpay, axis=1)
            if "score" in shop_features:
                x = np.insert(x, x.shape[1], shop_score, axis=1)
            if "comment" in shop_features:
                x = np.insert(x, x.shape[1], shop_comment, axis=1)
            if "level" in shop_features:
                x = np.insert(x, x.shape[1], shop_level, axis=1)
            shop_cate1 = oneshopinfo['cate1']
            if "cate1" in shop_features:
                shop_cate1_encoder = cate1_hot_encoder.transform(cate1_label_encoder.transform([shop_cate1]).reshape((-1, 1)))
                for i in range(shop_cate1_encoder.shape[1]):
                    x = np.insert(x,x.shape[1],shop_cate1_encoder[0][i],axis=1)
            '''商家信息添加完毕'''

            '''天气特征'''
            if weather:
                weathers = getOneWeekdayFomExtractedData(extractWeatherInfo(part_data, part_data.shape[0]-1, shop_city))
                x = np.append(x, weathers, axis=1)
            '''天气特征结束'''
            # for j in range(sameday_backNum):
            #     x.append(train_y[len(train_y) - (j+1)*7][0])
            # x = np.array(x).reshape((1, sameday_backNum))

            x = x_scaler.transform(x)
            predict = model.predict(x)
            '''将y还原'''
            if predict.ndim == 2:
                predict = y_scaler.inverse_transform(predict)[0][0]
            elif predict.ndim == 1:
                predict = y_scaler.inverse_transform(predict)[0]
            '''将y还原结束'''
            # print predict
            if(predict <= 0):
                predict == 1
            preficts.append(predict)
            part_data.set_value(part_data.shape[0]-1, "count", predict)

        preficts = (removeNegetive(toInt(np.array(preficts)))).astype(int)
        if preficts_all is None:
            preficts_all = preficts
        else:
            preficts_all = np.insert(preficts_all,preficts_all.shape[0],preficts,axis=0)

        if trainAsTest:
            last_14_real_y = (removeNegetive(toInt(np.array(last_14_real_y)))).astype(int)
            if real_all is None:
                real_all = last_14_real_y
            else:
                real_all = np.insert(real_all,real_all.shape[0],last_14_real_y,axis=0)
                # print preficts,last_14_real_y
            print str(j)+',score:', scoreoneshop(preficts, last_14_real_y)

    # preficts = np.array(preficts)
    shopids = shopids.tolist()
    if ignore_predict:
        for remove_id in ignore_shopids:
            try:
                shopids.remove(remove_id)
            except:
                pass

    preficts_all = preficts_all.reshape((len(shopids),14))
    if trainAsTest:
        real_all = real_all.reshape((len(shopids),14))
        preficts_all = np.concatenate((preficts_all, real_all), axis=1)



    preficts_all = np.insert(preficts_all, 0, shopids, axis=1)
    if saveFilePath is not None:
        if model_path is None:
            path = saveFilePath + "%dLast_%ds_%dd_%df_%d_%s_%d_%d_%d_%s_%dshops" \
                              % (last_N_days,sameday_backNum, day_back_num, train_x.shape[1], cate_level, cate_name
                                 ,  epochs, batch_size, h1_unit, h1_activation,len(shopids))
        else:
            import re
            r = re.compile(r"""/(\d+)last_(\d+)s_(\d+)d_(\d+)f_(\d+)_(\S+)_(\d+)_(\d+)_(\d+)_(\w+).json""")
            m = r.search(model_path)
            path = saveFilePath + "%dLast_%ds_%dd_%df_%d_%s_%d_%d_%d_%s_%dshops" \
                                  % (int(m.group(1)),int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6)
                                     ,  int(m.group(7)), int(m.group(8)), int(m.group(9)), m.group(10),len(shopids))
        if Augmented:
            path += "_augmented"
        if trainAsTest:
            path = path+"_train.csv"
        else:
            path = path + ".csv"

        print "save in :", path
        np.savetxt(path, preficts_all, fmt="%d", delimiter=",")
    return preficts_all


def predictAllShop_ANN2_HPS(all_data, trainAsTest=False, saveFilePath = None, featurePath = None,
                            cate_level = 0, cate_name = None, featureSavePath = None, needSaveFeature = False ,ignore_shopids = []
                            , needCV = False,
                            model_path = None):
    """
    通过gridsearch找超参数
    :param trainAsTest: 是否使用训练集后14天作为测试集
    :param model: 某个模型
    :param saveFilePath
    :param featurePath:
    :param cate_level:
    :param cate_name:
    :param featureSavePath:
    :param needSaveFeature:
    :param ignore_shopids:
    :param create_model_function:
    :param needCV
    :return:
    """

    verbose = 2
    last_N_days = 70
    #记录已经被忽略的商店数量
    ignores = 0
    shopids = None
    shop_need_to_predict = 2000
    if(cate_level is 0):
        shopids = np.arange(1, 1 + shop_need_to_predict, 1)
    else:
        shopids = Parameter.extractShopValueByCate(cate_level,cate_name)
    shop_info = pd.read_csv(Parameter.shopinfopath, names=["shopid", "cityname", "locationid", "perpay", "score", "comment", "level", "cate1", "cate2", "cate3"])

    weather = True
    weekOrWeekend = True
    day_back_num = 21
    sameday_backNum = 7
    week_backnum = 3
    other_features = [statistic_functon_mean,statistic_functon_median]
    other_features = []
    shop_features = ["perpay", "comment", "score", "level"]
    shop_features = []
    #是否是周末hot_encoder
    hot_encoder = onehot([[1], [0]])
    #类别1hot_encoder
    cate1_list = np.unique(shop_info['cate1'])
    cate1_label_encoder = labelEncoder(cate1_list)
    cate1_list2 = cate1_label_encoder.transform(cate1_list).reshape((-1, 1))
    cate1_hot_encoder = onehot(cate1_list2)

    if featurePath is None:
        all_x = None
        all_y = None
        for shopid in shopids:
            if shopid in ignore_shopids:
                print "ignore get train", shopid
                ignores += 1
                continue
            print "get ", shopid, " train"
            part_data = all_data[all_data.shopid == shopid]
            last_14_real_y = None
            # 取出一部分做训练集
            if trainAsTest: #使用训练集后14天作为测试集的话，训练集为前面部分
                last_14_real_y = part_data[len(part_data) - 14:]["count"].values
                part_data = part_data[0:len(part_data) - 14]
            # print last_14_real_y
            '''确定跳过前面多少天的数据'''
            skipNum = part_data.shape[0] - last_N_days
            if skipNum < 0:
                skipNum = 0
            train_x = None
            '''获取特征'''
            if sameday_backNum != 0: #sameday
                sameday = extractBackSameday(part_data, sameday_backNum, skipNum, nan_method_sameday_mean)
                train_x = getOneWeekdayFomExtractedData(sameday)
            if day_back_num != 0: #day
                if train_x is not None:
                    train_x = np.concatenate((train_x, getOneWeekdayFomExtractedData(extractBackDay(part_data, day_back_num, skipNum, nan_method_sameday_mean))),axis=1)
                else:
                    train_x = getOneWeekdayFomExtractedData(extractBackDay(part_data, day_back_num, skipNum, nan_method_sameday_mean))
            if weekOrWeekend: #weekOrWeekend
                ws = getOneWeekdayFomExtractedData(extractWorkOrWeekend(part_data, skipNum))
                train_x = np.concatenate((train_x, hot_encoder.transform(ws)), axis=1)


            count = extractCount(part_data, skipNum)
            train_y = getOneWeekdayFomExtractedData(count)
            for feature in other_features:
                value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, skipNum, nan_method_sameday_mean, feature))
                train_x = np.append(train_x, value, axis=1)

            '''添加商家信息'''
            # print train_x,train_x.shape
            index = shopid - 1
            oneshopinfo = shop_info.ix[index]
            shop_city = oneshopinfo['cityname']
            shop_perpay = oneshopinfo['perpay'] if not pd.isnull(oneshopinfo['perpay']) else 0
            shop_score = oneshopinfo['score'] if not pd.isnull(oneshopinfo['score']) else 0
            shop_comment = oneshopinfo['comment'] if not pd.isnull(oneshopinfo['comment']) else 0
            shop_level = oneshopinfo['level'] if not pd.isnull(oneshopinfo['level']) else 0
            shop_cate1 = oneshopinfo['cate1']
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=DeprecationWarning)
                shop_cate1_encoder = cate1_hot_encoder.transform(cate1_label_encoder.transform([shop_cate1]))
            if "perpay" in shop_features:
                train_x = np.insert(train_x, train_x.shape[1], shop_perpay, axis=1)
            if "score" in shop_features:
                train_x = np.insert(train_x, train_x.shape[1], shop_score, axis=1)
            if "comment" in shop_features:
                train_x = np.insert(train_x, train_x.shape[1], shop_comment, axis=1)
            if "level" in shop_features:
                train_x = np.insert(train_x, train_x.shape[1], shop_level, axis=1)
            if "cate1" in shop_features:
                for i in range(shop_cate1_encoder.shape[1]):
                    train_x = np.insert(train_x, train_x.shape[1], shop_cate1_encoder[0][i], axis=1)
            '''商家信息添加完毕'''

            '''天气特征'''
            if weather:
                weathers = getOneWeekdayFomExtractedData(extractWeatherInfo(part_data, skipNum, shop_city))
                train_x = np.append(train_x, weathers, axis=1)
            '''天气特征结束'''

            if all_x is None:
                all_x = train_x
                all_y = train_y
            else:
                all_x = np.insert(all_x,all_x.shape[0],train_x,axis=0)
                all_y = np.insert(all_y,all_y.shape[0],train_y,axis=0)

                # '''添加周几'''
                # extract_weekday = getOneWeekdayFomExtractedData(extractWeekday(part_data, skipNum))
                # train_x = np.append(train_x, extract_weekday, axis=1)
                # ''''''

                # train_x = train_x.reshape((train_x.shape[0],
                #                            train_x.shape[1], 1))
                # print model.get_weights()
                # part_counts = []
                # for i in range(7):
                #     weekday = i + 1
                #     part_count = getOneWeekdayFomExtractedData(count, weekday)
                #     part_counts.append(part_count)


        train_x = all_x
        train_y = all_y

        if needSaveFeature:
            featureAndLabel = np.concatenate((train_x, train_y), axis=1)
            flDF = pd.DataFrame(featureAndLabel)
            if featureSavePath is None:
                if trainAsTest:
                    featureSavePath = Parameter.projectPath + "lzj/train_feature/%dCatelevel_%sCatename_%dfeatures_%dSameday_%dDay_%dLast.csv" % (cate_level,cate_name,flDF.shape[1] - 1,sameday_backNum,day_back_num,last_N_days)
                else:
                    featureSavePath = Parameter.projectPath + "lzj/feature/%dCatelevel_%sCatename_%dfeatures_%dSameday_%dDay_%dLast.csv" % (cate_level, cate_name, flDF.shape[1] - 1, sameday_backNum, day_back_num, last_N_days)
            flDF.to_csv(featureSavePath)
    else:#有featurePath文件
        if trainAsTest:
            path = Parameter.projectPath + "lzj/train_feature/"+featurePath
        else:
            path = Parameter.projectPath + "lzj/feature/" + featurePath
        flDF = pd.read_csv(path, index_col=0)
        train_x = flDF.values[:, :-1]
        train_y = flDF.values[:, -1:]
        # print train_x
        # print train_y

    '''将t标准化'''
    x_scaler = MinMaxScaler().fit(train_x)
    y_scaler = MinMaxScaler().fit(train_y)
    train_x = x_scaler.transform(train_x)
    train_y = y_scaler.transform(train_y)
    '''标准化结束'''


    if model_path is None:
        if needCV:
            '''gridsearchCV'''
            # nb_epoch=rnn_epoch, batch_size=batch_size, verbose=verbose
            # input_dim, h1_unit = 16, optimizer = "adagrad", init = "normal"):
            input_dim = [train_x.shape[1]]
            h1_activation = ["relu"]
            h1_unit = [8,12,16,20]
            h2_unit=[2,4,6,8,10]
            model = KerasRegressor(build_fn=create_model2, verbose=verbose)
            batch_size = [3,5,7,10]
            epochs = [10,15,20,25,30,40]
            param_grid = dict(batch_size = batch_size, nb_epoch = epochs, h1_unit = h1_unit,h2_unit=h2_unit, input_dim = input_dim)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,scoring="neg_mean_squared_error")
            grid.refit = False
            grid_result = grid.fit(train_x, train_y)

            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            for params, mean_score, scores in grid_result.grid_scores_:
                print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

        if not needCV:
            input_dim = train_x.shape[1]
            h1_activation = "relu"
            h1_unit = 12
            h2_unit = 8
            batch_size = 3
            epochs = 80

        else:
            input_dim = train_x.shape[1]
            epochs = grid_result.best_params_['nb_epoch']
            batch_size = grid_result.best_params_['batch_size']
            h1_unit = grid_result.best_params_["h1_unit"]
            h2_unit = grid_result.best_params_["h2_unit"]
            h1_activation = "relu"

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        best_model = create_model2(input_dim=input_dim,h1_unit=h1_unit,h2_unit=h2_unit)
        hist = best_model.fit(train_x, train_y, verbose=verbose, batch_size=batch_size, nb_epoch=epochs,  validation_split = 0.1, callbacks=[early_stopping])
        print hist.history


        #保存模型
        if trainAsTest:
            model_save_path = Parameter.projectPath+"lzj/train_model/" + \
                              "%dlast_%ds_%dd_%df_%d_%s_%d_%d_%d_%s_%d_%s.json" \
                              % (last_N_days,sameday_backNum, day_back_num, train_x.shape[1], cate_level, cate_name
                                 , epochs, batch_size, h1_unit, h1_activation,h2_unit,h1_activation)
        else:
            model_save_path = Parameter.projectPath+"lzj/model/" + \
                              "%dlast_%ds_%dd_%df_%d_%s_%d_%d_%d_%s_%d_%s.json" \
                              % (last_N_days,sameday_backNum, day_back_num, train_x.shape[1], cate_level, cate_name
                                 ,  epochs, batch_size, h1_unit, h1_activation,h2_unit,h1_activation)
        print "model save in :", model_save_path
        saveModel(model_save_path, best_model)
    else:#model_path is not none
        best_model = getModel(model_path)

    format = "%Y-%m-%d"
    if trainAsTest:
        startTime = datetime.datetime.strptime("2016-10-18", format)
    else:
        startTime = datetime.datetime.strptime("2016-11-1", format)
    timedelta = datetime.timedelta(1)

    '''预测所有商家'''
    model = best_model
    preficts_all = None
    real_all = None
    for j in shopids:
        if j in ignore_shopids:
            print "ignore predict", j
            continue
        print "predict:", j
        preficts = []
        part_data = all_data[all_data.shopid == j]
        last_14_real_y = None

        if trainAsTest: #使用训练集后14天作为测试集的话，训练集为前面部分
            last_14_real_y = part_data[len(part_data) - 14:]["count"].values
            part_data = part_data[0:len(part_data) - 14]

        '''预测14天'''
        for i in range(14):
            currentTime = startTime + timedelta * i
            strftime = currentTime.strftime(format)
            # index = getWeekday(strftime) - 1
            # part_count = part_counts[index]
            #取前{sameday_backNum}周同一天的值为特征进行预测
            part_data = part_data.append({"count":0,"shopid":j,"time":strftime,"weekday":getWeekday(strftime)},ignore_index=True)
            x = None
            if sameday_backNum != 0:
                x = getOneWeekdayFomExtractedData(extractBackSameday(part_data,sameday_backNum,part_data.shape[0] - 1, nan_method_sameday_mean))
            if day_back_num != 0:
                if x is None:
                    x = getOneWeekdayFomExtractedData(extractBackDay(part_data,day_back_num,part_data.shape[0] -1 ,nan_method_sameday_mean))
                else:
                    x = np.concatenate((x, getOneWeekdayFomExtractedData(extractBackDay(part_data,day_back_num,part_data.shape[0] -1 ,nan_method_sameday_mean))),axis=1)
            if weekOrWeekend:
                x = np.concatenate((x, hot_encoder.transform(getOneWeekdayFomExtractedData(extractWorkOrWeekend(part_data,part_data.shape[0] - 1)))),axis=1)

            for feature in other_features:
                x_value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, part_data.shape[0]-1, nan_method_sameday_mean, feature))
                x = np.append(x, x_value, axis=1)
            # '''添加周几'''
            # x = np.append(x, getOneWeekdayFomExtractedData(extractWeekday(part_data, part_data.shape[0]-1)), axis=1)
            # ''''''

            '''添加商家信息'''
            index = j - 1
            oneshopinfo = shop_info.ix[index]
            shop_city = oneshopinfo["cityname"]
            shop_perpay = oneshopinfo['perpay'] if not pd.isnull(oneshopinfo['perpay']) else 0
            shop_score = oneshopinfo['score'] if not pd.isnull(oneshopinfo['score']) else 0
            shop_comment = oneshopinfo['comment'] if not pd.isnull(oneshopinfo['comment']) else 0
            shop_level = oneshopinfo['level'] if not pd.isnull(oneshopinfo['level']) else 0
            if "perpay" in shop_features:
                x = np.insert(x, x.shape[1], shop_perpay, axis=1)
            if "score" in shop_features:
                x = np.insert(x, x.shape[1], shop_score, axis=1)
            if "comment" in shop_features:
                x = np.insert(x, x.shape[1], shop_comment, axis=1)
            if "level" in shop_features:
                x = np.insert(x, x.shape[1], shop_level, axis=1)
            shop_cate1 = oneshopinfo['cate1']
            if "cate1" in shop_features:
                shop_cate1_encoder = cate1_hot_encoder.transform(cate1_label_encoder.transform([shop_cate1]).reshape((-1, 1)))
                for i in range(shop_cate1_encoder.shape[1]):
                    x = np.insert(x,x.shape[1],shop_cate1_encoder[0][i],axis=1)
            '''商家信息添加完毕'''

            '''天气特征'''
            if weather:
                weathers = getOneWeekdayFomExtractedData(extractWeatherInfo(part_data, part_data.shape[0]-1, shop_city))
                x = np.append(x, weathers, axis=1)
            x = x_scaler.transform(x)
            '''天气特征结束'''
            # for j in range(sameday_backNum):
            #     x.append(train_y[len(train_y) - (j+1)*7][0])
            # x = np.array(x).reshape((1, sameday_backNum))

            # print x
            # x = x.reshape(1, sameday_backNum, 1)
            predict = model.predict(x)
            if predict.ndim == 2:
                predict = y_scaler.inverse_transform(predict)[0][0]
            elif predict.ndim == 1:
                predict = y_scaler.inverse_transform(predict)[0]
            # print predict
            if(predict <= 0):
                predict == 1
            preficts.append(predict)
            part_data.set_value(part_data.shape[0]-1, "count", predict)

        preficts = (removeNegetive(toInt(np.array(preficts)))).astype(int)
        if preficts_all is None:
            preficts_all = preficts
        else:
            preficts_all = np.insert(preficts_all,preficts_all.shape[0],preficts,axis=0)

        if trainAsTest:
            last_14_real_y = (removeNegetive(toInt(np.array(last_14_real_y)))).astype(int)
            if real_all is None:
                real_all = last_14_real_y
            else:
                real_all = np.insert(real_all,real_all.shape[0],last_14_real_y,axis=0)
                # print preficts,last_14_real_y
            print str(j)+',score:', scoreoneshop(preficts, last_14_real_y)

    # preficts = np.array(preficts)
    preficts_all = preficts_all.reshape((len(shopids)-ignores,14))
    if trainAsTest:
        real_all = real_all.reshape((len(shopids) - ignores,14))
        preficts_all = np.concatenate((preficts_all, real_all), axis=1)
    shopids = shopids.tolist()
    for remove in ignore_shopids:
        try:
            shopids.remove(remove)
        except:
            pass
    preficts_all = np.insert(preficts_all, 0, shopids, axis=1)
    if saveFilePath is not None:
        path = saveFilePath + "%dLast_%ds_%dd_%df_%d_%s_%d_%d_%d_%s_%d_%s_%dshops" \
                              % (last_N_days,sameday_backNum, day_back_num, train_x.shape[1], cate_level, cate_name
                                 ,  epochs, batch_size, h1_unit, h1_activation,h2_unit,h1_activation,len(shopids)-ignores)
        if trainAsTest:
            path = path+"_train.csv"
        else:
            path = path + ".csv"

        print "save in :", path
        np.savetxt(path, preficts_all, fmt="%d", delimiter=",")
    return preficts_all




if __name__ == "__main__":

    payinfo = pd.read_csv(Parameter.payAfterGroupingAndRevision2AndTurncate_path)
    # fe_path = Parameter.projectPath + "lzj/train_feature/ann1_168.csv"
    # fe_path = "0Catelevel_NoneCatename_40features_7Sameday_21Day.csv"
    fe_path = None
    # fe_path="1Catelevel_美食Catename_10features_7Sameday_0Day_70Last.csv"
    # model_path = Parameter.projectPath + "lzj/train_model/70last_7s_0d_7f_1_美食_40_3_10_sigmoid.json"
    model_path = None
    predictAllShop_ANN1_HPS(payinfo, True, Parameter.projectPath + "result/ANN1_rt_hps",
                            fe_path, needSaveFeature=True, needCV=False,
                            cate_level=1, cate_name="超市便利店",
                            ignore_shopids= [23, 627, 749, 1269, 1218]
                            , Augmented=True, model_path=model_path, ignore_predict=False)

    #超市便利店 [23, 627, 749, 1269, 1875]
    # 美食 [5, 125, 284, 381, 411, 416, 434, 437, 444, 459, 470, 474, 501, 521, 524, 530, 619, 632, 654,
    #     659, 660, 683, 700, 721, 727, 735, 742, 752,
    #     768, 810, 956, 1050, 1058, 1100, 1107, 1145, 1163, 1185, 1214, 1241,
    #     1243, 1380, 1384, 1407, 1447, 1462, 1464, 1486, 1510, 1526, 1548, 1556, 1567, 1609,
    #     1650, 1681, 1716, 1730, 1747, 1769, 1803, 1831, 1835, 1856, 1858, 1893, 1918, 1968]