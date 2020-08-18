# import xgboost as xgb
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from .util import read_data_train, read_data_test, cal_train_future,cal_train_past, cal_test,read_data_df_test

def transforData(data):
    columns_origin = data.columns.tolist()
    #print(columns_origin)
    columns_names_X_new = ['1_1', '1_2', '1_3', '1_4', '2_1', '2_2', '2_3', '2_4', '3_1', '3_2', '3_3', '3_4', '4_1',
                           '4_2',
                           '4_3', '4_4',
                           '5_1', '5_2', '5_3', '5_4', '6_1', '6_2', '6_3', '6_4', '7_1', '7_2', '7_3', '7_4', '8_1',
                           '8_2',
                           '8_3', '8_4',
                           '9_1', '9_2', '9_3', '9_4', '10_1', '10_2', '10_3', '10_4', '11_1', '11_2', '11_3', '11_4',
                           '12_1',
                           '12_2', '12_3', '12_4']

    for i in range(2, len(columns_origin)-1):
        column_old = columns_origin[i]
        res = data[column_old].apply(lambda x: x.split(" "))
        #print(res)
        data[columns_names_X_new[(i-2)*4+0]] = [int(x[0]) for x in res.values]
        data[columns_names_X_new[(i-2)*4+1]] = [int(x[1]) for x in res.values]
        data[columns_names_X_new[(i-2)*4+2]] = [int(x[2]) for x in res.values]
        data[columns_names_X_new[(i-2)*4+3]] = [int(x[3]) for x in res.values]

    X_train = data.drop(['slice_1', 'slice_2', 'slice_3', 'slice_4', 'slice_5', 'slice_6', 'slice_7', 'slice_8', 'slice_9', 'slice_10', 'slice_11', 'slice_12','predict_6'], axis=1)
    #print(X_train.columns.tolist())
    return X_train




def week(time_year, time_month, time_day):
    #输出str
    week_ = datetime.datetime(time_year, time_month,time_day).strftime("%w")
    return int(week_) +1

def two_hour_mean(pre1,pre2,pre3,pre4,pre5,pre6,pre7,pre8,pre9,pre10,pre11,pre12):
    pre_mean = (pre1+pre2+pre3+pre4+pre5+pre6+pre7+pre8+pre9+pre10+pre11+pre12)/12
    return pre_mean

def csv_data_train(csv_paths_train):
    data = read_data_train(csv_paths_train)
    X_train = transforData(data)
    Y_train = pd.DataFrame()
    columns_names_Y_new = ['label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6']
    res_y = data['predict_6'].apply(lambda x: x.split(" "))
    for i in range(0, 6):
        Y_train[columns_names_Y_new[i]] = [int(x[i]) for x in res_y.values]

    # 获取 日期数据 的年、月、日、时、分
    X_train['time_slice'] = pd.to_datetime(X_train['time'], format='%Y/%m/%d %H:%M')
    X_train['time_year'] = X_train['time_slice'].dt.year
    X_train['time_month'] = X_train['time_slice'].dt.month
    X_train['time_day'] = X_train['time_slice'].dt.day
    X_train['time_hour'] = X_train['time_slice'].dt.hour
    X_train['time_minute'] = X_train['time_slice'].dt.minute
    X_train['time_week'] = X_train.apply(lambda x: week(x['time_year'], x['time_month'], x['time_day']), axis=1)
    X_train['two_hour_mean'] = X_train.apply(lambda x: two_hour_mean(x['1_3'],x['2_3'],x['3_3'],x['4_3'],x['5_3'],
                                                                     x['6_3'],x['7_3'],x['8_3'],x['9_3'],x['10_3'],
                                                                     x['11_3'],x['12_3']), axis=1)

    # print(X_train['two_hour_mean'])
    # 获取 格子id，按照后九位去中间一位j进行划分
    list = []
    for id in X_train['id']:
        id_split = id[7:11] + id[12:16]
        # print(id_split)
        list.append(int(id_split))
    X_train['id_slice'] = list
    X_train = X_train.drop(['time', 'id','time_slice'], axis=1)
    # print(X_train.columns.tolist())

    return X_train, Y_train

def csv_data_test(csv_path_test):
    df_test = read_data_df_test(csv_path_test)
    X_test = transforData(df_test)

    # 获取 日期数据 的年、月、日、时、分
    X_test['time_slice'] = pd.to_datetime(X_test['time'], format='%Y/%m/%d %H:%M')
    X_test['time_year'] = X_test['time_slice'].dt.year
    X_test['time_month'] = X_test['time_slice'].dt.month
    X_test['time_day'] = X_test['time_slice'].dt.day
    X_test['time_hour'] = X_test['time_slice'].dt.hour
    X_test['time_minute'] = X_test['time_slice'].dt.minute
    X_test['time_week'] = X_test.apply(lambda x: week(x['time_year'], x['time_month'], x['time_day']), axis=1)
    X_test['two_hour_mean'] = X_test.apply(lambda x: two_hour_mean(x['1_3'],x['2_3'],x['3_3'],x['4_3'],x['5_3'],
                                                                     x['6_3'],x['7_3'],x['8_3'],x['9_3'],x['10_3'],
                                                                     x['11_3'],x['12_3']), axis=1)

    # print(X_train['two_hour_mean'])
    # 获取 格子id，按照后九位去中间一位j进行划分
    list = []
    for id in X_test['id']:
        id_split = id[7:11] + id[12:16]
        # print(id_split)
        list.append(int(id_split))
    X_test['id_slice'] = list
    X_test = X_test.drop(['time', 'id','time_slice'], axis=1)
    # print(X_test.columns.tolist())

    return X_test

# def addTimeFeature(X):
#
#
#
#     # 获取 日期数据 的年、月、日、时、分
#     X['time_slice'] = pd.to_datetime(X['time'], format='%Y/%m/%d %H:%M')
#     #X['time_year'] = X['time_slice'].dt.year
#     #X['time_month'] = X['time_slice'].dt.month
#     X['time_day'] = X['time_slice'].dt.day
#     X['time_hour'] = X['time_slice'].dt.hour
#     X['time_minute'] = X['time_slice'].dt.minute
#     # 获取 格子id，按照后九位去中间一位j进行划分
#     list = []
#     for id in X['id']:
#         id_split = id[7:11] + id[12:16]
#         # print(id_split)
#         list.append(id_split)
#     X['id_slice'] = list
#     X_new = X.drop(['time', 'id', 'time_slice'], axis=1)
#     # print(X_train)
#     #print(X_new.columns.tolist())
#     return X_new

#
# def csv_data_train(csv_paths_train):
#     data = read_data_train(csv_paths_train)
#     X_train = transforData(data)
#     Y_train = pd.DataFrame()
#     columns_names_Y_new = ['label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6']
#     res_y = data['predict_6'].apply(lambda x: x.split(" "))
#     for i in range(0, 6):
#         Y_train[columns_names_Y_new[i]] = [int(x[i]) for x in res_y.values]
#
#     X_train_new = addTimeFeature(X_train)
#
#     #print(Y_train.columns.tolist)
#
#     return X_train_new, Y_train
#



#
# def csv_data_test(csv_path_test):
#     #time, id, slice_1, slice_2, slice_3, slice_4, slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12 = read_data_df_test(csv_path_test)
#     df_test = read_data_df_test(csv_path_test)
#     X_test = transforData(df_test)
#     #X_test_new = addTimeFeature(X_test)
#     return X_test_new