import xgboost as xgb
import numpy as np
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def saveModel(model):
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d-%H-%M-%S')
    model_name = 'model/xgb-' + time_str + ".model"
    model.save_model(model_name)











def trainModel(X, Y, random_seed=12):
    params = {
        'booster': 'gbtree',  # gbtree used
        'objective': 'binary:logistic',
        'early_stopping_rounds': 50,
        'scale_pos_weight': 0.63,  # 正样本权重
        'eval_metric': 'auc',
        'gamma': 0,
        'max_depth': 5,
        # 'lambda': 550,
        'subsample': 0.6,
        'colsample_bytree': 0.9,
        'min_child_weight': 1,
        'eta': 0.02,
        'seed': random_seed,
        'nthread': 3,
        'silent': 0
    }
    #print(X_train)



    # dtest = xgb.DMatrix(X_test.drop(['Kind'], axis=1))
    # dtrain = xgb.DMatrix(X_train.drop(['Kind'], axis=1), Y_train.drop(['Kind'], axis=1))

    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')

    #XGBoost训练过程
    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    # print(np.shape(X.values))
    # print(np.shape(Y.values))
    predict_Y = []
    for train_index, test_index in skf.split(X, X['time_hour']):
        X_train, Y_train = X.iloc[train_index], Y.iloc[train_index]
        X_test, Y_test = X.iloc[test_index], Y.iloc[test_index]
        model.fit(X_train.values, Y_train['label_1'].values)
        # 对测试集进行预测
        predict_label = model.predict(X_test.values)
        #print(np.shape(predict_label))
        predict_Y.extend(predict_label)

    mse_split = mean_squared_error(Y['label_1'].values, predict_Y)
    mae = mean_absolute_error(Y['label_1'].values, predict_Y)
    print("mean_squared_error:", mse_split)
    print("mean_absolute_error", mae )

    return model





