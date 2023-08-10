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
        # 数据复原方法
        with open('data.json', 'r') as f:
            json_data = f.read()
        # 将JSON字符串转换为字典
        my_dict = json.loads(json_data)
        # 输出字典
        # print(my_dict)

        one_data = all_data.iloc[data_index].tolist()
        real_out_data = origin_data.iloc[data_index].tolist()[-2:]

        # 初始化神经网络模型
        predict_net = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )
        file_path = "model_pytorch/model_430.pth"
        loaded_model = torch.load(file_path)
        predict_net.eval()
        input_data = torch.tensor([one_data[:16]], dtype=torch.float32)
        out_data = torch.tensor([one_data[-2:]], dtype=torch.float32)
        pre_out = loaded_model(input_data)
        pre_out = pre_out.detach()
        pre_out[0][0] = pre_out[0][0] * my_dict['one_time_flow']['std'] + my_dict['one_time_flow']['mean']
        pre_out[0][1] = pre_out[0][1] * my_dict['water_ele']['std'] + my_dict['water_ele']['mean']
        pre_out = list(pre_out.squeeze().numpy())
        pre_out[0] = round(pre_out[0], 0)
        pre_out[1] = round(pre_out[1], 1)
        print("predict_data:", pre_out)
        print("real_out_data:", real_out_data)
        print("................................................................")
