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
import random

if __name__ == '__main__':
    all_data = pd.read_csv("train_deal_data.csv")
    origin_data = pd.read_csv("../data/train_data.csv")
    data_max_index = len(all_data) - 1

    for i in range(0, 100):
        data_index = random.randint(0, data_max_index)
        with open('data.json', 'r') as f:
            json_data = f.read()
        # 将JSON字符串转换为字典
        my_dict = json.loads(json_data)
        one_data = all_data.iloc[data_index].tolist()
        real_out_data = origin_data.iloc[data_index].tolist()[-2:]

        out_data = one_data[-2:]
        out_data[0] = out_data[0] * my_dict['one_time_flow']['std'] + my_dict['one_time_flow']['mean']
        out_data[1] = out_data[1] * my_dict['water_ele']['std'] + my_dict['water_ele']['mean']

        out_data[0] = round(out_data[0], 0)
        out_data[1] = round(out_data[1], 1)

        print("out_data:", out_data)
        print("real_out_data:", real_out_data)
        print("................................................................")
