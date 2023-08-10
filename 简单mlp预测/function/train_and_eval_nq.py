# 找到（频率，流量）对应的功率
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import os
from collections import deque
import shutil
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
import json

if __name__ == '__main__':
    print('Starting training')
    # #处理数据
    # all_data = pd.read_csv("../data/train_data.csv")
    # new_data = all_data.drop(all_data.columns[[0]], axis=1)
    # new_data.columns = ["one_beng_pressure","two_beng_pressure","three_beng_pressure","four_beng_pressure",
    #                 "five_beng_pressure","six_beng_pressure","seven_beng_pressure","eight_beng_pressure",
    #
    #                 "one_beng_fre","two_beng_fre","three_beng_fre","four_beng_fre",
    #                 "five_beng_fre","six_beng_fre","seven_beng_fre","eight_beng_fre",
    #
    #                 "one_time_flow", "water_ele"]
    #
    # new_data.to_csv("../data/train_data_no_index.csv", index=False)



    all_data = pd.read_csv("../data/train_data_no_index.csv")
    normalization_parameters = {}

    after_deal_data = pd.DataFrame()

    for column in all_data.columns:
        mean_value = all_data[column].mean()
        std_value = all_data[column].std()
        normalization_parameters[column] = {'mean': mean_value, 'std': std_value}
        after_deal_data[f'{column}_normalized'] = (all_data[column] - mean_value) / std_value

    after_deal_data.to_csv("train_deal_data.csv",index=False)
    with open('data.json', 'w') as json_file:
        json.dump(normalization_parameters, json_file, indent=4)

    print("save_success")
