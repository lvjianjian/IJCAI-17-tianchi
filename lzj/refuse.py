#encoding=utf-8
import Parameter
import pandas as pd
from cjx_predict import scoreoneshop,score
import numpy as np
def computeScore(filePath, scoreFilePath,  threshold =0.06, needRefuseDataPath = None, refuseDataPath = None, refuseDataSavePath = None):

    train_predict = np.loadtxt(filePath, dtype=int, delimiter=",")
    shopids = train_predict.take(0,axis=1).tolist()
    predicts = np.ndarray(0)
    reals = np.ndarray(0)
    good = []
    bad = []
    scores = []
    for k in range(len(shopids)):
        id = shopids[k]
        predict = train_predict[k][1:15]
        real = train_predict[k][15:29]
        predicts = np.append(predicts,predict)
        reals = np.append(reals,real)
        score_one = scoreoneshop(predict, real)
        print id, ":", score_one
        if(score_one<threshold):
            good.append(id)
        else:
            bad.append(id)
        scores.append(score_one)
    print "last score:", score(predicts,reals)
    print "good", good, len(good)
    print "bad", bad, len(bad)

    if scoreFilePath is not None:
        result = np.reshape(scores, (len(shopids), 1))
        result = np.insert(result,0,shopids,axis=1)
        np.savetxt(scoreFilePath, (result), delimiter=",", fmt="%.6f")

    if needRefuseDataPath is not None:
        needRefuseData = np.loadtxt(needRefuseDataPath, dtype=int, delimiter=",")
        refuseData = np.loadtxt(refuseDataPath, dtype=int, delimiter=",")
        refuse_data = np.zeros((len(shopids), 14))
        for i in range(len(shopids)):
            shopid = i+1
            if shopid in good:
                value = needRefuseData[i][1:15]
            elif shopid in bad:
                value = refuseData[i][1:15]
            refuse_data[i] = value
        refuse_data = np.insert(refuse_data,0,shopids,axis=1).astype(int)
        np.savetxt(refuseDataSavePath, refuse_data, delimiter=",", fmt='%d')


def computeScoreByOrigin(filePath, scoreFilePath,  threshold =0.06, needRefuseDataPath = None, refuseDataPath = None, refuseDataSavePath = None):
    import Parameter
    origin = pd.read_csv(Parameter.payAfterGrouping_path)
    reals = np.ndarray(0)
    train_predict = np.loadtxt(filePath, dtype=int, delimiter=",")
    shopids = train_predict.take(0, axis=1).tolist()
    for shopid in shopids:
        part_data = origin[origin.shopid == shopid]
        last_14_real_y = None
        # 取出一部分做训练集
        last_14_real_y = part_data[len(part_data) - 14:]["count"].values
        reals = np.append(reals,last_14_real_y)
    reals = reals.reshape((len(shopids),14))
    predicts = np.ndarray(0)
    good = []
    bad = []
    scores = []
    for k in range(len(shopids)):
        id = shopids[k]
        predict = train_predict[k][1:15]
        predicts = np.append(predicts,predict)
        score_one = scoreoneshop(predict, reals[k])
        print id, ":", score_one
        if(score_one<threshold):
            good.append(id)
        else:
            bad.append(id)
        scores.append(score_one)
    print "last score:", score(predicts, reals.reshape(14 * train_predict.shape[0]))
    print "good", good, len(good)
    print "bad", bad, len(bad)

    if scoreFilePath is not None:
        result = np.reshape(scores, (len(shopids), 1))
        result = np.insert(result,0,shopids,axis=1)
        np.savetxt(scoreFilePath, (result), delimiter=",", fmt="%.6f")

    if needRefuseDataPath is not None:
        needRefuseData = np.loadtxt(needRefuseDataPath, dtype=int, delimiter=",")
        refuseData = np.loadtxt(refuseDataPath, dtype=int, delimiter=",")
        refuse_data = np.zeros((2000, 14))
        for i in range(2000):
            shopid = i+1
            try:
                index = shopids.index(shopid)
            except:
                index = -1
            if shopid in good:
                value = needRefuseData[index][1:15]
            elif shopid in bad:
                value = refuseData[i][1:15]
            else:
                value = refuseData[i][1:15]
            refuse_data[i] = value
        refuse_data = np.insert(refuse_data,0,range(1, 2001, 1),axis=1).astype(int)
        np.savetxt(refuseDataSavePath, refuse_data, delimiter=",", fmt='%d')


def merge(paths,base_result_path,final_result_save_path,threshold = -1):
    if threshold != -1:
        bad_idss = []
        origin = pd.read_csv(Parameter.payAfterGrouping_path)
        for path in paths:
            if "train" not in path:
                path = path.replace(".csv", "_train.csv")
            reals = np.ndarray(0)
            train_predict = np.loadtxt(path, dtype=int, delimiter=",")
            shopids = train_predict.take(0, axis=1).tolist()
            for shopid in shopids:
                part_data = origin[origin.shopid == shopid]
                last_14_real_y = None
                # 取出一部分做训练集
                last_14_real_y = part_data[len(part_data) - 14:]["count"].values
                reals = np.append(reals,last_14_real_y)
            reals = reals.reshape((len(shopids),14))
            predicts = np.ndarray(0)
            bad = []
            for k in range(len(shopids)):
                id = shopids[k]
                predict = train_predict[k][1:15]
                predicts = np.append(predicts,predict)
                score_one = scoreoneshop(predict, reals[k])
                if(score_one>threshold):
                    bad.append(id)
            bad_idss.append(bad)

    train = False
    if "train" in paths[0]:
        train = True
    datas = []
    shopids = []
    indexes = []
    shops = 0
    insert_index = 0
    for path in paths:
        loadtxt = np.loadtxt(path, delimiter=",", dtype=int)
        datas.append(loadtxt)
        shopids.append(loadtxt.take(0,axis=1))
        indexes.append(0)
        shops += loadtxt.shape[0]
    if base_result_path is not  None:
        base_data = np.loadtxt(base_result_path, delimiter=",", dtype=int)

    remove = 0
    if bad_idss is not None:
        for badids in bad_idss:
            remove += len(badids)
    print "bad number:", remove
    if base_result_path is not None:
        final_result = np.ndarray((2000, 15))
    else:
        final_result = np.ndarray((shops - remove, 15))
    for shopid in range(1, 2001, 1):
        insert = False
        for j in range(len(datas)):
            if not insert:
                if shopid in bad_idss[j]:
                    indexes[j] += 1
                    continue
                if shopid in shopids[j]:
                    if not train:
                        final_result[insert_index] = datas[j][indexes[j]]
                    else:
                        final_result[insert_index] = datas[j][indexes[j]][0:15]
                    indexes[j] += 1
                    insert = True
                    insert_index += 1

        if not insert:
            if base_result_path is not None:
                final_result[insert_index] = base_data[shopid-1]
                insert_index += 1
    np.savetxt(final_result_save_path, final_result, fmt="%d", delimiter=",")


def weightMerge(paths,weights,save_path = None):
    datas = []
    for path in paths:
        datas.append(np.loadtxt(path,delimiter=",",dtype=int))
    shopids = datas[0][:,:1]
    result = datas[0] * weights[0]
    for i in range(1,len(weights),1):
        result += datas[i] * weights[i]
    final_result = result[:,1:].astype(int)
    final_result = np.concatenate((shopids,final_result),axis=1)
    if save_path is not None:
        np.savetxt(save_path,final_result,fmt="%d",delimiter=",")
    return final_result

if __name__ == "__main__":
    # computeScore("../result/RF_9f_train.csv",None,0.07,Parameter.projectPath+"result/RF_9f.csv",Parameter.nearestmean_len3_weightmedianmore_v4_csv,"refuse_RF_9f.csv")
    # computeScoreByOrigin("../result/ANN3_rt_train_20_16_relu.csv", None, 0.1)
    # computeScore("../result/ANN3_rt_train_20_16_relu.csv", None, 0.1)
    from sys import argv
    try:
        path = argv[1]
    except:
        path = None
    if path is None:
        path = "RF_rt_hps60Last_7s_21d_28f_1_美食_10_2_6_auto_1415shops_1time_train" + ".csv"
    thre = 0.2
    computeScoreByOrigin(Parameter.projectPath + "result/"+path, None, thre)

    # computeScoreByOrigin("/home/zhongjianlv/IJCAI/lzj/final_result/CNN_rt_hps60Last_0s_21d_21f_1_超市便利店_40_3_20_sigmoid_574shops_train_1time.csv", None, thre)
    # merge([Parameter.projectPath + "result/AdaBoost_rt_hps60Last_7s_21d_28f_1_美食_75_100_0_-1_exponential_1415shops_1time.csv",
    #        Parameter.projectPath + "result/AdaBoost_rt_hps60Last_7s_21d_28f_1_超市便利店_75_100_0_-1_exponential_579shops_1time.csv"],
    #       Parameter.fs182good, "merge_Adaboost_超市_美食.csv",threshold=thre)
    #Parameter.fs182good
    # computeScoreByOrigin(Parameter.projectPath + "lzj/merge_Adaboost_超市_美食_train.csv", None, 0.2)

    # weightMerge(["/home/zhongjianlv/IJCAI/result/AdaBoost_rt_hps60Last_0s_21d_21f_1_超市便利店_100_0_-1_exponential_579shops_1time.csv",
    #              "/home/zhongjianlv/IJCAI/result/AdaBoost_rt_hps60Last_7s_0d_7f_1_超市便利店_75_0_-1_exponential_579shops_1time.csv"],
    #             [0.5,0.5],save_path="/home/zhongjianlv/IJCAI/result/AdaBoost_rt_hps60Last_7s_21d_28f_1_超市便利店_75_100_0_-1_exponential_579shops_1time.csv")