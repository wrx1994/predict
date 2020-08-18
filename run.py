# -*- encoding:utf-8 -*-
import os
import argparse
import pandas as pd
from method.pre import csv_data_train, csv_data_test
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_importance
# from sklearn.preprocessing import Imputer换成下边的就可以
from sklearn.impute import SimpleImputer

def get_parser():
    parser = argparse.ArgumentParser(description="Run in csvs")
    parser.add_argument("--csvTrain", default="/Users/ruoxi/didi/在弦上/在弦上2020实战2班3班/在弦上2020实战2班3班/实战3班/train/", help="path to csv folder")
    parser.add_argument("--csvTest", default="/Users/ruoxi/didi/在弦上/在弦上2020实战2班3班/在弦上2020实战2班3班/实战3班/test1/", help="path to csv folder")

    return parser

def getAllCSV(folderPath):
    # 初始化一个列表
    outputLst = []
    # 输出路径下文件列表，无序的。排序：video_list.sort()
    csv_list = os.listdir(folderPath)
    # idx为索引，video为列表元素
    for idx, csv in enumerate(csv_list):
        # 路径拼接，函数会自动加/
        csv_path = os.path.join(folderPath, csv)
        # 在列表末尾添加新元素
        outputLst.append(csv_path)
    return outputLst

def trainandTest(X_train, Y_train, X_test):
        # XGBoost训练过程
        model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
        model.fit(X_train, Y_train)

        # 对测试集进行预测
        ans = model.predict(X_test)

        ans_len = len(ans)
        id_list = X_train['1']
        data_arr = []
        for row in range(0, ans_len):
            data_arr.append([id_list[row], ans[row]])
        np_data = np.array(data_arr)

        # 写入文件
        pd_data = pd.DataFrame(np_data, columns=['id', 'y'])
        # print(pd_data)
        pd_data.to_csv('submit.csv', index=None)


if __name__ == "__main__":
    args = get_parser().parse_args()
    csv_paths_train = getAllCSV(args.csvTrain)
    csv_paths_test = getAllCSV(args.csvTest)
    for csv_path_train in csv_paths_train:
        print(csv_path_train)
        # X为过去值，Y为未来预测值
        # x_train为dict，y_train为list
        x_train,y_train = csv_data_train(csv_path_train)
        print(x_train)
        print(y_train)
        for csv_path_test in csv_paths_test:
            print(csv_path_test)
            # x_test为dict
            x_test = csv_data_test(csv_path_test)
            print(x_test)
            # trainandTest(x_train, y_train,x_test)