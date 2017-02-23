#encoding=utf-8

import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder

from FeatureExtractor import *
from cjx_predict import scoreoneshop
from lv import removeNegetive,toInt


def predictOneShop_LR(shopid, all_data, trainAsTest=False):
    """
    用LR预测某一个商店
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
    # print model.get_weights()
    # part_counts = []
    # for i in range(7):
    #     weekday = i + 1
    #     part_count = getOneWeekdayFomExtractedData(count, weekday)
    #     part_counts.append(part_count)

    # print getrefcount(model)

    model = LinearRegression()
    model.fit(train_x,train_y)

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


def predictOneShop_RR(shopid, all_data, trainAsTest=False):
    """
    用RidgeRegression预测某一个商店
    :param shopid: 预测商店id
    :param trainAsTest: 是否使用训练集后14天作为测试集
    :return:
    """
    alpha = 0.5


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
    # print model.get_weights()
    # part_counts = []
    # for i in range(7):
    #     weekday = i + 1
    #     part_count = getOneWeekdayFomExtractedData(count, weekday)
    #     part_counts.append(part_count)

    # print getrefcount(model)

    model = Ridge(alpha)
    model.fit(train_x,train_y)

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


def predictOneShop_RCV(shopid, all_data, trainAsTest=False):
    #用交叉验证测试给定的alpha,并挑选最合适的
    model1 = RidgeCV(alphas=np.arange(0.1,1,0.05))
    return predictOneShop_model(shopid, all_data, trainAsTest, model1)

def predictOneShop_RF(shopid,all_data,trainAsTest=False):
    model = RandomForestRegressor(max_depth=28, n_estimators=128, min_samples_split=10)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=DeprecationWarning)
        return predictOneShop_model(shopid,all_data,trainAsTest,model)
    # print model

def predictOneShop_model(shopid, all_data, trainAsTest=False, model = None):
    """
    用RidgeCV预测某一个商店
    :param shopid: 预测商店id
    :param trainAsTest: 是否使用训练集后14天作为测试集
    :param model: 某个模型
    :return:
    """

    if (model is None):
        raise Exception("model is None ")
        return

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
    # print model.get_weights()
    # part_counts = []
    # for i in range(7):
    #     weekday = i + 1
    #     part_count = getOneWeekdayFomExtractedData(count, weekday)
    #     part_counts.append(part_count)

    # print getrefcount(model)

    model.fit(train_x,train_y)

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
        if predict.ndim == 2:
            predict = y_scaler.inverse_transform(predict)[0][0]
        elif predict.ndim == 1:
            predict = y_scaler.inverse_transform(predict)[0]

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


def predictAllShop_model_together(all_data, trainAsTest=False, model = None, saveFilePath = None, featurePath = None):
    """
    使用所有商家所有数据训练,预测所有商店
    :param trainAsTest: 是否使用训练集后14天作为测试集
    :param model: 某个模型
    :return:
    """
    shop_need_to_predict = 2000
    if (model is None):
        raise Exception("model is None")
        return

    shop_info = pd.read_csv(Parameter.shopinfopath,names=["shopid","cityname","locationid","perpay","score","comment","level","cate1","cate2","cate3"])

    sameday_backNum = 3
    week_backnum = 3
    other_features = [statistic_functon_mean,statistic_functon_median]
    '''将cate1 onehot'''
    cate = shop_info['cate1'].tolist()
    cate_dup = set(cate)
    cates = []
    for i in range(len(cate_dup)):
        cates.append([i])
    hot_encoder = OneHotEncoder().fit(cates)
    dicts = dict(zip(cate_dup, range(len(cate_dup))))
    cate_num = []
    for c in cate:
        cate_num.append([dicts[c]])
    '''cate1 onehot finish'''
    if featurePath is None:

        all_x = None
        all_y = None
        for shopid in range(1, 1+shop_need_to_predict, 1):
            print "get " , shopid, " train"
            part_data = all_data[all_data.shopid == shopid]
            last_14_real_y = None
            # 取出一部分做训练集
            if trainAsTest: #使用训练集后14天作为测试集的话，训练集为前面部分
                last_14_real_y = part_data[len(part_data) - 14:]["count"].values
                part_data = part_data[0:len(part_data) - 14]
            # print last_14_real_y
            skipNum = part_data.shape[0] - 128
            if skipNum < 0:
                skipNum = 0

            sameday = extractBackSameday(part_data, sameday_backNum, skipNum, nan_method_sameday_mean)
            count = extractCount(part_data, skipNum)
            train_x = getOneWeekdayFomExtractedData(sameday)
            train_y = getOneWeekdayFomExtractedData(count)
            for feature in other_features:
                value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, skipNum, nan_method_sameday_mean, feature))
                train_x = np.append(train_x, value, axis=1)

            '''添加商家信息'''
            # print train_x,train_x.shape
            index = shopid - 1
            oneshopinfo = shop_info.ix[index]
            shop_perpay = oneshopinfo['perpay'] if not pd.isnull(oneshopinfo['perpay']) else 0
            shop_score = oneshopinfo['score'] if not pd.isnull(oneshopinfo['score']) else 0
            shop_comment = oneshopinfo['comment'] if not pd.isnull(oneshopinfo['comment']) else 0
            shop_level = oneshopinfo['level'] if not pd.isnull(oneshopinfo['level']) else 0
            shop_cate1 = oneshopinfo['cate1']
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=DeprecationWarning)
                shop_cate1_encoder = hot_encoder.transform([dicts[shop_cate1]]).toarray()
            train_x = np.insert(train_x,train_x.shape[1],shop_perpay,axis=1)
            train_x = np.insert(train_x,train_x.shape[1],shop_score,axis=1)
            train_x = np.insert(train_x,train_x.shape[1],shop_comment,axis=1)
            train_x = np.insert(train_x,train_x.shape[1],shop_level,axis=1)
            for i in range(shop_cate1_encoder.shape[1]):
                train_x = np.insert(train_x,train_x.shape[1],shop_cate1_encoder[0][i],axis=1)
            '''商家信息添加完毕'''

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
        featureAndLabel = np.concatenate((train_x,train_y),axis=1)
        flDF = pd.DataFrame(featureAndLabel, columns=["sameday1","sameday2","sameday3","week_mean1","week_mean2","week_mean3","week_median1","week_median2","week_median3","perpay","score","comment","level","cate1_1","cate1_2","cate1_3","cate1_4","cate1_5","cate1_6","label"])
        if trainAsTest:
            flDF.to_csv("train_feature/19f.csv")
        else:
            flDF.to_csv("feature/19f.csv")
    else:#有featurePath文件
        flDF = pd.read_csv(featurePath,index_col=0)
        train_x = flDF.values[:,:-1]
        train_y = flDF.values[:,-1:]
        # print train_x
        # print train_y

    '''将t标准化'''
    x_scaler = MinMaxScaler().fit(train_x)
    y_scaler = MinMaxScaler().fit(train_y)
    train_x = x_scaler.transform(train_x)
    train_y = y_scaler.transform(train_y)
    '''标准化结束'''


    model.fit(train_x,train_y)

    format = "%Y-%m-%d"
    if trainAsTest:
        startTime = datetime.datetime.strptime("2016-10-18", format)
    else:
        startTime = datetime.datetime.strptime("2016-11-1", format)
    timedelta = datetime.timedelta(1)


    '''预测所有商家'''
    preficts_all = None
    real_all = None
    for j in range(1, 1 + shop_need_to_predict, 1):
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
            x = getOneWeekdayFomExtractedData(extractBackSameday(part_data,sameday_backNum,part_data.shape[0] - 1, nan_method_sameday_mean))
            for feature in other_features:
                x_value = getOneWeekdayFomExtractedData(extractBackWeekValue(part_data, week_backnum, part_data.shape[0]-1, nan_method_sameday_mean, feature))
                x = np.append(x, x_value, axis=1)
            # '''添加周几'''
            # x = np.append(x, getOneWeekdayFomExtractedData(extractWeekday(part_data, part_data.shape[0]-1)), axis=1)
            # ''''''
            '''添加商家信息'''
            index = j - 1
            oneshopinfo = shop_info.ix[index]
            shop_perpay = oneshopinfo['perpay'] if not pd.isnull(oneshopinfo['perpay']) else 0
            shop_score = oneshopinfo['score'] if not pd.isnull(oneshopinfo['score']) else 0
            shop_comment = oneshopinfo['comment'] if not pd.isnull(oneshopinfo['comment']) else 0
            shop_level = oneshopinfo['level'] if not pd.isnull(oneshopinfo['level']) else 0
            shop_cate1 = oneshopinfo['cate1']
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=DeprecationWarning)
                shop_cate1_encoder = hot_encoder.transform([dicts[shop_cate1]]).toarray()
            x = np.insert(x,x.shape[1],shop_perpay,axis=1)
            x = np.insert(x,x.shape[1],shop_score,axis=1)
            x = np.insert(x,x.shape[1],shop_comment,axis=1)
            x = np.insert(x,x.shape[1],shop_level,axis=1)
            for i in range(shop_cate1_encoder.shape[1]):
                x = np.insert(x,x.shape[1],shop_cate1_encoder[0][i],axis=1)
            '''商家信息添加完毕'''

            x = x_scaler.transform(x)
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
    preficts_all = preficts_all.reshape((shop_need_to_predict,14))
    if trainAsTest:
        real_all = real_all.reshape((shop_need_to_predict,14))
        preficts_all = np.concatenate((preficts_all,real_all), axis=1)
    preficts_all = np.insert(preficts_all, 0, range(1, shop_need_to_predict+1, 1), axis=1)
    if saveFilePath is not None:
        np.savetxt(saveFilePath,preficts_all,fmt="%d",delimiter=",")
    return preficts_all

def predictAllShop_RF_together(all_data, trainAsTest=False, saveFilePath = None):
    model = RandomForestRegressor(max_depth=28, n_estimators=256, min_samples_split=1000)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        if trainAsTest:
            return predictAllShop_model_together(all_data,trainAsTest,model,saveFilePath,"train_feature/19f.csv")
        else:
            return predictAllShop_model_together(all_data,trainAsTest,model,saveFilePath,"feature/19f.csv")


if __name__ == "__main__":
    import Parameter
    payinfo = pd.read_csv(Parameter.payAfterGroupingAndRevision2AndTurncate_path)
    # predict_all(payinfo, Parameter.projectPath + "result/RF_9f.csv", False, [1, 2000], predictOneShop_RF)

    # predictAllShop_model_together(payinfo, True, RidgeCV(np.arange(0.1, 1.0, 0.1))
    #                               , Parameter.projectPath+"result/all_RF.csv")
    predictAllShop_RF_together(payinfo, False, Parameter.projectPath+"result/all_RF.csv")
    # shopid = 23
    # predictOneShop_LR(shopid, payinfo, True)
    # predictOneShop_RR(shopid, payinfo, True)
    # predictOneShop_RCV(shopid, payinfo, True)
    # predictOneShop_RF(shopid,payinfo,True)