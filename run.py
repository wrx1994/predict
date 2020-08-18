# -*- encoding:utf-8 -*-
import os
import argparse
import pandas as pd
import numpy as np
from method.pre import csv_data_train, csv_data_test
# from sklearn.preprocessing import Imputer换成下边的就可以
import xgboost as xgb
from xgb import trainandTest



def get_parser():
    parser = argparse.ArgumentParser(description="Run in csvs")
    parser.add_argument("--csvTrain", default="/Users/didi/project/predict/train/", help="path to csv folder")
    parser.add_argument("--csvTest", default="/Users/didi/project/predict/test1/", help="path to csv folder")

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


def test_model():
    w = pd.DataFrame({17, 10, 100, 100, 64.639999, 7, 0, 0, 0, 0, 7, 6, 13, 12, 11.545455, 11.454545, 7, 6, 13, 12, 11.545455, 11.454545, 610, 610, 5, 5, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      0.66290897, 0.94605702, 1.7, 900, 1, 1, 0, 11625, 8576, 1, 1, 1, -1, -1, -1, -1, -1, -1})
    xx = xgb.DMatrix(w)
    bst_new = xgb.Booster({'nthread': 4})  # init model
    bst_new.load_model("model/xgboost_v1.3.model")
    bst_new.predict(xx)



if __name__ == "__main__":

    #test_model()
    args = get_parser().parse_args()
    csv_paths_train = getAllCSV(args.csvTrain)
    csv_paths_test = getAllCSV(args.csvTest)
    X_train = pd.DataFrame()
    Y_train = pd.DataFrame()
    X_test = pd.DataFrame()
    cnt1=0;cnt2 =0
    for csv_path_train in csv_paths_train:
        print(csv_path_train)
        if(cnt1==1):
            break
        cnt1+=1
        x_train, y_train = csv_data_train(csv_path_train)
        X_train = pd.concat([X_train, x_train], axis=0)
        Y_train = pd.concat([Y_train, y_train], axis=0)
        #print(X_train)
    for csv_path_test in csv_paths_test:
        print(csv_path_test)
        if (cnt2 == 1):
            break
        cnt2 += 1
        x_test = csv_data_test(csv_path_test)
        X_test = pd.concat([X_test, x_test], axis=0)
    #print(np.shape(X_train))
    print(X_train.columns.tolist)
    trainandTest(X_train, Y_train, X_test)