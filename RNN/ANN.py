#encoding=utf-8
from FeatureExtractor import *
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,LSTM
from keras.models import Sequential,Merge
from keras.optimizers import  RMSprop,SGD
import datetime
from cjx_predict import scoreoneshop,score
from lv import toInt,removeNegetive
import Parameter
import gc
import numpy as np
from SRN import predict_all,predict_all_getbest
from SRN import my_loss
# np.random.seed(228)



def predictOneShop_ANN(shopid, all_data, trainAsTest=False):
    """
    用ANN预测某一个商店,一个隐藏层,一个网络
    :param shopid: 预测商店id
    :param trainAsTest: 是否使用训练集后14天作为测试集
    :return:
    """
    part_data = all_data[all_data.shopid == shopid]
    last_14_real_y = None
    # 取出一部分做训练集
    if trainAsTest: #使用训练集后14天作为测试集的话，训练集为前面部分
        last_14_real_y = part_data[len(part_data) - 14:]["count"].values
        part_data = part_data[0:len(part_data) - 14]
    # print last_14_real_y
    verbose = 2
    rnn_nb_epoch = 20
    skipNum = 28
    sameday_backNum = 3
    week_backnum = 3
    sameday = extractBackSameday(part_data, sameday_backNum, skipNum, nan_method_sameday_mean)
    count = extractCount(part_data, skipNum)
    train_x = getOneWeekdayFomExtractedData(sameday)
    train_y = getOneWeekdayFomExtractedData(count)
    other_features = [statistic_functon_mean,statistic_functon_median]
    for feature in other_features:
        value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, skipNum, nan_method_sameday_mean, feature))
        train_x = np.append(train_x, value, axis=1)

    # '''添加周几'''
    # extract_weekday = getOneWeekdayFomExtractedData(extractWeekday(part_data, skipNum))
    # train_x = np.append(train_x, extract_weekday, axis=1)
    # ''''''

    '''将t标准化'''
    x_scaler = MinMaxScaler().fit(train_x)
    y_scaler = MinMaxScaler().fit(train_y)
    train_x = x_scaler.transform(train_x)
    train_y = y_scaler.transform(train_y)
    '''标准化结束'''
    # train_x = train_x.reshape((train_x.shape[0],
    #                            train_x.shape[1], 1))
    model = Sequential()
    # print getrefcount(model)
    model.add(Dense(32, input_dim=train_x.shape[1], activation="tanh")) #sigmoid
    # print getrefcount(model)
    model.add(Dense(1, activation='linear'))
    #, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)
    # print getrefcount(model)
    # 设置优化器（除了学习率外建议保持其他参数不变）
    # sgd=SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer="sgd")
    # print model.summary()
    # print getrefcount(model)
    # print model.summary()
    model.fit(train_x, train_y, nb_epoch=rnn_nb_epoch, batch_size=1, verbose=verbose)
    # print model.get_weights()
    # part_counts = []
    # for i in range(7):
    #     weekday = i + 1
    #     part_count = getOneWeekdayFomExtractedData(count, weekday)
    #     part_counts.append(part_count)

    # print getrefcount(model)
    format = "%Y-%m-%d"
    if trainAsTest:
        startTime = datetime.datetime.strptime("2016-10-18", format)
    else:
        startTime = datetime.datetime.strptime("2016-11-1", format)
    timedelta = datetime.timedelta(1)
    preficts = []
    for i in range(14):
        currentTime = startTime + timedelta * i
        strftime = currentTime.strftime(format)
        # index = getWeekday(strftime) - 1
        # part_count = part_counts[index]
        #取前{sameday_backNum}周同一天的值为特征进行预测
        part_data = part_data.append({"count":0,"shopid":shopid,"time":strftime,"weekday":getWeekday(strftime)},ignore_index=True)
        x = getOneWeekdayFomExtractedData(extractBackSameday(part_data,sameday_backNum,part_data.shape[0] - 1, nan_method_sameday_mean))
        for feature in other_features:
            x_value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, part_data.shape[0]-1, nan_method_sameday_mean, feature))
            x = np.append(x, x_value, axis=1)
        # '''添加周几'''
        # x = np.append(x, getOneWeekdayFomExtractedData(extractWeekday(part_data, part_data.shape[0]-1)), axis=1)
        # ''''''

        x = x_scaler.transform(x)
        # for j in range(sameday_backNum):
        #     x.append(train_y[len(train_y) - (j+1)*7][0])
        # x = np.array(x).reshape((1, sameday_backNum))

        # print x
        # x = x.reshape(1, sameday_backNum, 1)
        predict = model.predict(x)
        predict = y_scaler.inverse_transform(predict)[0][0]
        if(predict <= 0):
            predict == 1
        preficts.append(predict)
        part_data.set_value(part_data.shape[0]-1, "count", predict)
        # preficts.append(predict)
        # part_counts[index] = np.append(part_count, predict).reshape((part_count.shape[0] + 1, 1))
    preficts = (removeNegetive(toInt(np.array(preficts)))).astype(int)
    # preficts = np.array(preficts)
    if trainAsTest:
        last_14_real_y = (removeNegetive(toInt(np.array(last_14_real_y)))).astype(int)
        # print preficts,last_14_real_y
        print str(shopid)+',score:', scoreoneshop(preficts, last_14_real_y)
    return [preficts, last_14_real_y]

def predictOneShop_ANN2(shopid, all_data, trainAsTest=False, best_model=None):
    """
    用ANN预测某一个商店,一个隐藏层,一个网络
    :param shopid: 预测商店id
    :param trainAsTest: 是否使用训练集后14天作为测试集
    :return:
    """
    # if trainAsTest is False:
    #     raise Exception("trainAsTest should be True, not support False")
    #     return


    skipNum = 28
    sameday_backNum = 3
    week_backnum = 3

    part_data = all_data[all_data.shopid == shopid]
    last_14_real_y = None
    # 取出一部分做训练集
    if trainAsTest: #使用训练集后14天作为测试集的话，训练集为前面部分
        last_14_real_y = part_data[len(part_data) - 14:]["count"].values
        part_data = part_data[0:len(part_data) - 14]
    # print last_14_real_y
    verbose = 0
    rnn_nb_epoch = 10
    sameday = extractBackSameday(part_data, sameday_backNum, skipNum, nan_method_sameday_mean)
    count = extractCount(part_data, skipNum)
    train_x = getOneWeekdayFomExtractedData(sameday)
    train_y = getOneWeekdayFomExtractedData(count)
    other_features = [statistic_functon_mean,statistic_functon_median]
    for feature in other_features:
        value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, skipNum, nan_method_sameday_mean, feature))
        train_x = np.append(train_x, value, axis=1)

    # '''添加周几'''
    # extract_weekday = getOneWeekdayFomExtractedData(extractWeekday(part_data, skipNum))
    # train_x = np.append(train_x, extract_weekday, axis=1)
    # ''''''

    '''将t标准化'''
    x_scaler = MinMaxScaler().fit(train_x)
    y_scaler = MinMaxScaler().fit(train_y)
    train_x = x_scaler.transform(train_x)
    train_y = y_scaler.transform(train_y)
    '''标准化结束'''
    # train_x = train_x.reshape((train_x.shape[0],  train_x.shape[1], 1))
    #
    if best_model is None:
        model = Sequential()
        # print getrefcount(model)
        model.add(Dense(32, input_dim=train_x.shape[1], activation="tanh")) #sigmoid
        # print getrefcount(model)
        model.add(Dense(1, activation='linear'))
        #, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)
        # print getrefcount(model)
        # 设置优化器（除了学习率外建议保持其他参数不变）
        # sgd = SGD(lr=0.005)
        model.compile(loss="mse", optimizer="sgd")
        # print model.summary()
        # print getrefcount(model)
        # print model.summary()
        model.fit(train_x, train_y, nb_epoch=rnn_nb_epoch, batch_size=1, verbose=verbose)
    else:
        model = best_model
        # model.fit(train_x, train_y, nb_epoch=rnn_nb_epoch, batch_size=1, verbose=verbose)
# print model.get_weights()
    # part_counts = []
    # for i in range(7):
    #     weekday = i + 1
    #     part_count = getOneWeekdayFomExtractedData(count, weekday)
    #     part_counts.append(part_count)
    # print getrefcount(model)
    format = "%Y-%m-%d"
    if trainAsTest:
        startTime = datetime.datetime.strptime("2016-10-18", format)
    else:
        startTime = datetime.datetime.strptime("2016-11-1", format)

    timedelta = datetime.timedelta(1)
    preficts = []
    for i in range(14):
        currentTime = startTime + timedelta * i
        strftime = currentTime.strftime(format)
        # index = getWeekday(strftime) - 1
        # part_count = part_counts[index]
        #取前{sameday_backNum}周同一天的值为特征进行预测
        part_data = part_data.append({"count":0, "shopid":shopid, "time":strftime, "weekday":getWeekday(strftime)},ignore_index=True)
        x = getOneWeekdayFomExtractedData(extractBackSameday(part_data,sameday_backNum,part_data.shape[0] - 1, nan_method_sameday_mean))
        for feature in other_features:
            x_value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, part_data.shape[0]-1, nan_method_sameday_mean, feature))
            x = np.append(x, x_value, axis=1)
        # '''添加周几'''
        # x = np.append(x, getOneWeekdayFomExtractedData(extractWeekday(part_data, part_data.shape[0]-1)), axis=1)
        # ''''''

        x = x_scaler.transform(x)
        # for j in range(sameday_backNum):
        #     x.append(train_y[len(train_y) - (j+1)*7][0])
        # x = np.array(x).reshape((1, sameday_backNum))

        # print x
        # x = x.reshape(1, sameday_backNum, 1)
        predict = model.predict(x)
        predict = y_scaler.inverse_transform(predict)[0][0]
        if(predict <= 0):
            predict == 1
        preficts.append(predict)
        part_data.set_value(part_data.shape[0]-1, "count", predict)
        # preficts.append(predict)
        # part_counts[index] = np.append(part_count, predict).reshape((part_count.shape[0] + 1, 1))
    preficts = (removeNegetive(toInt(np.array(preficts)))).astype(int)
    # preficts = np.array(preficts)
    if trainAsTest:
        last_14_real_y = (removeNegetive(toInt(np.array(last_14_real_y)))).astype(int)
        # print preficts,last_14_real_y
        print str(shopid)+',score:', scoreoneshop(preficts, last_14_real_y)
    return [preficts, last_14_real_y, model]



def predictOneShop_2ANNrefuse_2(shopid, all_data, trainAsTest=False):
    """
    用ANN预测某一个商店,2个网络分别模拟近期趋势和中期趋势,隐藏层合并,1层隐藏层
    结果不太稳定，应该是因为近期数据用ANN来做不太好
    :param shopid: 预测商店id
    :param trainAsTest: 是否使用训练集后14天作为测试集
    :return:
    """
    part_data = all_data[all_data.shopid == shopid]
    last_14_real_y = None
    # 取出一部分做训练集
    if trainAsTest: #使用训练集后14天作为测试集的话，训练集为前面部分
        last_14_real_y = part_data[len(part_data) - 14:]["count"].values
        part_data = part_data[0:len(part_data) - 14]
    # print last_14_real_y
    verbose = 2
    rnn_nb_epoch = 25
    skipNum = 28
    day_backNum = 7
    sameday_backNum = 3
    week_backnum = 3
    learnrate = 0.01
    sameday = extractBackSameday(part_data, sameday_backNum, skipNum, nan_method_sameday_mean)
    day = extractBackDay(part_data,day_backNum,skipNum,nan_method_sameday_mean)
    count = extractCount(part_data, skipNum)
    train_x = getOneWeekdayFomExtractedData(sameday)
    train_x2 = getOneWeekdayFomExtractedData(day)
    train_y = getOneWeekdayFomExtractedData(count)
    other_features = [statistic_functon_mean,statistic_functon_median]
    # other_features = []
    for feature in other_features:
        value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, skipNum, nan_method_sameday_mean, feature))
        train_x = np.append(train_x, value, axis=1)

    # '''添加周几'''
    # extract_weekday = getOneWeekdayFomExtractedData(extractWeekday(part_data, skipNum))
    # train_x = np.append(train_x, extract_weekday, axis=1)
    # ''''''

    '''将t标准化'''
    x_scaler = MinMaxScaler().fit(train_x)
    x2_scaler = MinMaxScaler().fit(train_x2)
    y_scaler = MinMaxScaler().fit(train_y)
    train_x = x_scaler.transform(train_x)
    train_x2 = x2_scaler.transform(train_x2)
    train_y = y_scaler.transform(train_y)
    '''标准化结束'''
    # train_x = train_x.reshape((train_x.shape[0],
    #                            train_x.shape[1], 1))
    model1 = Sequential()
    model2 = Sequential()
    final_model = Sequential()
    # print getrefcount(model1)
    model1.add(Dense(32, input_dim=train_x.shape[1], activation="sigmoid")) #sigmoid
    model1.add(Dense(1, activation='linear'))


    '''近期趋势'''
    model2.add(Dense(32,input_dim=train_x2.shape[1],activation="sigmoid"))
    model2.add(Dense(1, activation='sigmoid'))

    final_model.add(Merge([model1, model2],mode="concat",concat_axis=1))
    final_model.add(Dense(1, activation='sigmoid'))

    #, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)
    # print getrefcount(model1)
    # 设置优化器（除了学习率外建议保持其他参数不变）
    rms=RMSprop(lr=0.05)
    # sgd=SGD(lr=0.1, momentum=0.9, nesterov=True)
    final_model.compile(loss="mse", optimizer="sgd")
    print final_model.summary()
    # print model1.summary()
    # print getrefcount(model1)
    # print model1.summary()
    final_model.fit([train_x, train_x2], train_y, nb_epoch=rnn_nb_epoch, batch_size=1, verbose=verbose)
    # print model1.get_weights()
    # part_counts = []
    # for i in range(7):
    #     weekday = i + 1
    #     part_count = getOneWeekdayFomExtractedData(count, weekday)
    #     part_counts.append(part_count)

    # print getrefcount(model1)
    format = "%Y-%m-%d"
    if trainAsTest:
        startTime = datetime.datetime.strptime("2016-10-18", format)
    else:
        startTime = datetime.datetime.strptime("2016-11-1", format)
    timedelta = datetime.timedelta(1)
    preficts = []
    for i in range(14):
        currentTime = startTime + timedelta * i
        strftime = currentTime.strftime(format)
        # index = getWeekday(strftime) - 1
        # part_count = part_counts[index]
        #取前{sameday_backNum}周同一天的值为特征进行预测
        part_data = part_data.append({"count":0, "shopid":shopid, "time":strftime, "weekday":getWeekday(strftime)}, ignore_index=True)
        x = getOneWeekdayFomExtractedData(extractBackSameday(part_data,sameday_backNum,part_data.shape[0] - 1, nan_method_sameday_mean))
        x2 = getOneWeekdayFomExtractedData(extractBackDay(part_data,day_backNum,part_data.shape[0]-1,nan_method_sameday_mean))
        for feature in other_features:
            x_value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, part_data.shape[0]-1, nan_method_sameday_mean, feature))
            x = np.append(x, x_value, axis=1)
        # '''添加周几'''
        # x = np.append(x, getOneWeekdayFomExtractedData(extractWeekday(part_data, part_data.shape[0]-1)), axis=1)
        # ''''''

        x = x_scaler.transform(x)
        x2 = x2_scaler.transform(x2)
        # for j in range(sameday_backNum):
        #     x.append(train_y[len(train_y) - (j+1)*7][0])
        # x = np.array(x).reshape((1, sameday_backNum))

        # print x
        # x = x.reshape(1, sameday_backNum, 1)
        predict = final_model.predict([x,x2])
        predict = y_scaler.inverse_transform(predict)[0][0]
        if(predict <= 0):
            predict == 1
        preficts.append(predict)
        part_data.set_value(part_data.shape[0]-1, "count", predict)
        # preficts.append(predict)
        # part_counts[index] = np.append(part_count, predict).reshape((part_count.shape[0] + 1, 1))
    preficts = (removeNegetive(toInt(np.array(preficts)))).astype(int)
    # preficts = np.array(preficts)
    if trainAsTest:
        last_14_real_y = (removeNegetive(toInt(np.array(last_14_real_y)))).astype(int)
        # print preficts,last_14_real_y
        print str(shopid)+',score:', scoreoneshop(preficts, last_14_real_y)
    return [preficts, last_14_real_y]

def predictOneShop_ANN_LSTM(shopid, all_data, trainAsTest=False):
    """
    用ANN预测某一个商店,2个网络分别模拟近期趋势和中期趋势,隐藏层合并,1层隐藏层,近期趋势用LSTM,速度慢,而且效果有时候也不好
    :param shopid: 预测商店id
    :param trainAsTest: 是否使用训练集后14天作为测试集
    :return:
    """
    part_data = all_data[all_data.shopid == shopid]
    last_14_real_y = None
    # 取出一部分做训练集
    if trainAsTest: #使用训练集后14天作为测试集的话，训练集为前面部分
        last_14_real_y = part_data[len(part_data) - 14:]["count"].values
        part_data = part_data[0:len(part_data) - 14]
    # print last_14_real_y
    verbose = 2
    rnn_nb_epoch = 10
    skipNum = 28
    day_backNum = 7
    sameday_backNum = 3
    week_backnum = 3
    learnrate = 0.01
    sameday = extractBackSameday(part_data, sameday_backNum, skipNum, nan_method_sameday_mean)
    day = extractBackDay(part_data,day_backNum,skipNum,nan_method_sameday_mean)
    count = extractCount(part_data, skipNum)
    train_x = getOneWeekdayFomExtractedData(sameday)
    train_x2 = getOneWeekdayFomExtractedData(day)
    train_y = getOneWeekdayFomExtractedData(count)
    other_features = [statistic_functon_mean,statistic_functon_median]
    # other_features = []
    for feature in other_features:
        value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, skipNum, nan_method_sameday_mean, feature))
        train_x = np.append(train_x, value, axis=1)

    # '''添加周几'''
    # extract_weekday = getOneWeekdayFomExtractedData(extractWeekday(part_data, skipNum))
    # train_x = np.append(train_x, extract_weekday, axis=1)
    # ''''''

    '''将t标准化'''
    x_scaler = MinMaxScaler().fit(train_x)
    x2_scaler = MinMaxScaler().fit(train_x2)
    y_scaler = MinMaxScaler().fit(train_y)
    train_x = x_scaler.transform(train_x)
    train_x2 = x2_scaler.transform(train_x2)
    train_x2 = train_x2.reshape((train_x2.shape[0],
                                 train_x2.shape[1], 1))
    train_y = y_scaler.transform(train_y)
    '''标准化结束'''
    # train_x = train_x.reshape((train_x.shape[0],
    #                            train_x.shape[1], 1))
    model1 = Sequential()
    model2 = Sequential()
    final_model = Sequential()
    # print getrefcount(model1)
    model1.add(Dense(32, input_dim=train_x.shape[1], activation="sigmoid")) #sigmoid
    # model1.add(Dense(1, activation='linear'))


    '''近期趋势'''
    model2.add(LSTM(32, input_shape=(train_x2.shape[1],train_x2.shape[2]), activation="sigmoid"))


    final_model.add(Merge([model1, model2],mode="concat",concat_axis=1))
    final_model.add(Dense(1, activation='linear'))

    #, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)
    # print getrefcount(model1)
    # 设置优化器（除了学习率外建议保持其他参数不变）
    rms=RMSprop(lr=0.05)
    # sgd=SGD(lr=0.1, momentum=0.9, nesterov=True)
    final_model.compile(loss="mse", optimizer=rms)
    print final_model.summary()
    # print model1.summary()
    # print getrefcount(model1)
    # print model1.summary()
    final_model.fit([train_x, train_x2], train_y, nb_epoch=rnn_nb_epoch, batch_size=1, verbose=verbose)
    # print model1.get_weights()
    # part_counts = []
    # for i in range(7):
    #     weekday = i + 1
    #     part_count = getOneWeekdayFomExtractedData(count, weekday)
    #     part_counts.append(part_count)

    # print getrefcount(model1)
    format = "%Y-%m-%d"
    if trainAsTest:
        startTime = datetime.datetime.strptime("2016-10-18", format)
    else:
        startTime = datetime.datetime.strptime("2016-11-1", format)
    timedelta = datetime.timedelta(1)
    preficts = []
    for i in range(14):
        currentTime = startTime + timedelta * i
        strftime = currentTime.strftime(format)
        # index = getWeekday(strftime) - 1
        # part_count = part_counts[index]
        #取前{sameday_backNum}周同一天的值为特征进行预测
        part_data = part_data.append({"count":0, "shopid":shopid, "time":strftime, "weekday":getWeekday(strftime)}, ignore_index=True)
        x = getOneWeekdayFomExtractedData(extractBackSameday(part_data,sameday_backNum,part_data.shape[0] - 1, nan_method_sameday_mean))
        x2 = getOneWeekdayFomExtractedData(extractBackDay(part_data,day_backNum,part_data.shape[0]-1,nan_method_sameday_mean))
        for feature in other_features:
            x_value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, part_data.shape[0]-1, nan_method_sameday_mean, feature))
            x = np.append(x, x_value, axis=1)
        # '''添加周几'''
        # x = np.append(x, getOneWeekdayFomExtractedData(extractWeekday(part_data, part_data.shape[0]-1)), axis=1)
        # ''''''

        x = x_scaler.transform(x)
        x2 = x2_scaler.transform(x2)
        x2 = x2.reshape((x2.shape[0],x2.shape[1],1))
        # for j in range(sameday_backNum):
        #     x.append(train_y[len(train_y) - (j+1)*7][0])
        # x = np.array(x).reshape((1, sameday_backNum))

        # print x
        # x = x.reshape(1, sameday_backNum, 1)
        predict = final_model.predict([x,x2])
        predict = y_scaler.inverse_transform(predict)[0][0]
        if(predict <= 0):
            predict == 1
        preficts.append(predict)
        part_data.set_value(part_data.shape[0]-1, "count", predict)
        # preficts.append(predict)
        # part_counts[index] = np.append(part_count, predict).reshape((part_count.shape[0] + 1, 1))
    preficts = (removeNegetive(toInt(np.array(preficts)))).astype(int)
    # preficts = np.array(preficts)
    if trainAsTest:
        last_14_real_y = (removeNegetive(toInt(np.array(last_14_real_y)))).astype(int)
        # print preficts,last_14_real_y
        print str(shopid)+',score:', scoreoneshop(preficts, last_14_real_y)
    return [preficts, last_14_real_y]



if __name__ == "__main__":
    from sys import argv
    pay_info = pd.read_csv(Parameter.payAfterGroupingAndRevision2AndTurncate_path, index_col=0)
    if(len(argv) != 1):
        startid = int(argv[1])
        endid = int(argv[2])
        predict_all_getbest(pay_info, Parameter.projectPath + "result/ANN_9f_best_%d.csv" % startid, False, [startid, endid], predictOneShop_ANN2, 10)
    else:
        # predict_all_getbest(pay_info, Parameter.projectPath + "result/ANN_9f_best_%d.csv", False, [6, 6], predictOneShop_ANN2, 10)
        print predictOneShop_ANN(1528, pay_info, True)
