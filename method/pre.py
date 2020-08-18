# import xgboost as xgb
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from .util import read_data_train, read_data_test, cal_past, cal_future

def csv_data_train(csv_paths_train):
    time, id, slice_1, slice_2, slice_3, slice_4, slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, \
        slice_12, predict_6 = read_data_train(csv_paths_train)
    df = pd.DataFrame()
    # 由行到列
    df['time'] = np.array(time).T
    df['id'] = np.array(id).T
    df['slice_1'] = np.array(slice_1).T
    df['slice_2'] = np.array(slice_2).T
    df['slice_3'] = np.array(slice_3).T
    df['slice_4'] = np.array(slice_4).T
    df['slice_5'] = np.array(slice_5).T
    df['slice_6'] = np.array(slice_6).T
    df['slice_7'] = np.array(slice_7).T
    df['slice_8'] = np.array(slice_8).T
    df['slice_9'] = np.array(slice_9).T
    df['slice_10'] = np.array(slice_10).T
    df['slice_11'] = np.array(slice_11).T
    df['slice_12'] = np.array(slice_12).T
    df['predict_6'] = np.array(predict_6).T

    #输出的time，id是字符串，其余是int类型
    input_past = df.apply(lambda x: cal_past(x['time'],x['id'],x['slice_1'],x['slice_2'],x['slice_3'],x['slice_4'],x['slice_5'],
                           x['slice_6'],x['slice_7'],x['slice_8'],x['slice_9'],x['slice_10'],x['slice_11'],
                           x['slice_12']),axis=1)
    input_future = df.apply(lambda x: cal_future(x['predict_6']),axis=1)

    return input_past, input_future


def csv_data_test(csv_path_test):
    time, id, slice_1, slice_2, slice_3, slice_4, slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12 = read_data_test(csv_path_test)
    df = pd.DataFrame()
    df['time'] = np.array(time).T
    df['id'] = np.array(id).T
    df['slice_1'] = np.array(slice_1).T
    df['slice_2'] = np.array(slice_2).T
    df['slice_3'] = np.array(slice_3).T
    df['slice_4'] = np.array(slice_4).T
    df['slice_5'] = np.array(slice_5).T
    df['slice_6'] = np.array(slice_6).T
    df['slice_7'] = np.array(slice_7).T
    df['slice_8'] = np.array(slice_8).T
    df['slice_9'] = np.array(slice_9).T
    df['slice_10'] = np.array(slice_10).T
    df['slice_11'] = np.array(slice_11).T
    df['slice_12'] = np.array(slice_12).T
    input_past = df.apply(lambda x: cal_past(x['time'], x['id'], x['slice_1'], x['slice_2'], x['slice_3'], x['slice_4'], x['slice_5'],
                           x['slice_6'], x['slice_7'], x['slice_8'], x['slice_9'], x['slice_10'], x['slice_11'],
                           x['slice_12']), axis=1)
    return df['time'],df['id'],input_past
