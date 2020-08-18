import pandas as pd

def read_data_df_train(input_file):
    # sep=',' ，分割
    name_list = ["time", "id", "slice_1", "slice_2", "slice_3", "slice_4", "slice_5", "slice_6", "slice_7", "slice_8", "slice_9", "slice_10", "slice_11", "slice_12", "predict_6"]
    df = pd.read_csv(input_file, sep = ',', header=None, names=name_list)
    return df

def read_data_train(input_file):
    datas = read_data_df_train(input_file)
    return [list(datas[tmp]) for tmp in ["time", "id", "slice_1", "slice_2", "slice_3", "slice_4", "slice_5", "slice_6", "slice_7",
        "slice_8", "slice_9", "slice_10", "slice_11", "slice_12", "predict_6"]]

def read_data_df_test(input_file):
    # sep=',' ，分割
    name_list = ["time", "id", "slice_1", "slice_2", "slice_3", "slice_4", "slice_5", "slice_6", "slice_7", "slice_8", "slice_9", "slice_10", "slice_11", "slice_12", "predict_6"]
    df = pd.read_csv(input_file, sep = ',', header=None, names=name_list)
    return df

def read_data_test(input_file):
    datas = read_data_df_test(input_file)
    return [list(datas[tmp]) for tmp in ["time", "id", "slice_1", "slice_2", "slice_3", "slice_4", "slice_5", "slice_6", "slice_7",
        "slice_8", "slice_9", "slice_10", "slice_11", "slice_12"]]


def cal_past(time, id, slice_1,slice_2,slice_3,slice_4,slice_5,slice_6,slice_7,slice_8,slice_9,slice_10,slice_11,slice_12):
    type = ['1','2','3','4','5','6']
    slice_1_value = slice_1.split(' ')
    slice_2_value = slice_2.split(' ')
    slice_3_value = slice_3.split(' ')
    slice_4_value = slice_4.split(' ')
    slice_5_value = slice_5.split(' ')
    slice_6_value = slice_6.split(' ')
    slice_7_value = slice_7.split(' ')
    slice_8_value = slice_8.split(' ')
    slice_9_value = slice_9.split(' ')
    slice_10_value = slice_10.split(' ')
    slice_11_value = slice_11.split(' ')
    slice_12_value = slice_12.split(' ')

    past_value_1 = [int(slice_1_value[0]),int(slice_2_value[0]),int(slice_3_value[0]),int(slice_4_value[0]),int(slice_5_value[0]),int(slice_6_value[0]),int(slice_7_value[0]),int(slice_8_value[0]),int(slice_9_value[0]),int(slice_10_value[0]),int(slice_11_value[0]),int(slice_12_value[0])]
    past_value_2 = [int(slice_1_value[1]),int(slice_2_value[1]),int(slice_3_value[1]),int(slice_4_value[1]),int(slice_5_value[1]),int(slice_6_value[1]),int(slice_7_value[1]),int(slice_8_value[1]),int(slice_9_value[1]),int(slice_10_value[1]),int(slice_11_value[1]),int(slice_12_value[1])]
    past_value_3 = [int(slice_1_value[2]),int(slice_2_value[2]),int(slice_3_value[2]),int(slice_4_value[2]),int(slice_5_value[2]),int(slice_6_value[2]),int(slice_7_value[2]),int(slice_8_value[2]),int(slice_9_value[2]),int(slice_10_value[2]),int(slice_11_value[2]),int(slice_12_value[2])]
    past_value_4 = [int(slice_1_value[3]),int(slice_2_value[3]),int(slice_3_value[3]),int(slice_4_value[3]),int(slice_5_value[3]),int(slice_6_value[3]),int(slice_7_value[3]),int(slice_8_value[3]),int(slice_9_value[3]),int(slice_10_value[3]),int(slice_11_value[3]),int(slice_12_value[3])]
    list_value = [time, id,past_value_1,past_value_2,past_value_3,past_value_4]
    #由两个列表转换为一个dict，type为key的列表，list_value为value列表
    input_past = dict(zip(type,list_value))
    return input_past

def cal_future(predict_6):
    predict_6_value = predict_6.split(' ')
    input_future = [int(predict_6_value[0]), int(predict_6_value[1]), int(predict_6_value[2]), int(predict_6_value[3]),int(predict_6_value[4]), int(predict_6_value[5])]
    return input_future
