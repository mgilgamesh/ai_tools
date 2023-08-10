# 对单泵进行建模,找到（压力，频率）对应的流量
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

if __name__ == "__main__":

    one_beng_data = pd.read_csv("one_beng_data.csv")
    # 获取频率，压力 二元组 对应的流量
    evl_data_len = 400
    train_data_len = len(one_beng_data) - 400
    eval_data = one_beng_data.sample(n=evl_data_len)
    train_data = one_beng_data.sample(n=train_data_len)

    # 原始输入数据集
    data_input_pandas = train_data[['1号单泵频率', '1号泵口表显压力']]
    data_output_pandas = train_data[['1号单泵瞬时流量']]

    # 目标数据归一化
    data_output_pandas = np.array(data_output_pandas)

    scaler = StandardScaler()
    scaler.fit(data_output_pandas)
    joblib.dump(scaler, 'scaler.pkl')
    data_output = scaler.transform(data_output_pandas)
    # 初始化原始输入数据
    data_input = torch.from_numpy(np.array(data_input_pandas)).float()
    data_output = torch.from_numpy(np.array(data_output)).float()

    torch_dataset = data.TensorDataset(data_input, data_output)
    # 把dataset放入DataLoader
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=300,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）
        num_workers=2  # 多少线程来读取数据
    )

    # 机器学习超参数及神经网络模型
    lr = 1e-3
    K_epochs = 100

    predict_net = nn.Sequential(
        nn.BatchNorm1d(2),
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
    )
    optimizer_adam = torch.optim.Adam([
        {'params': predict_net.parameters(), 'lr': lr}
    ])

    criterion = nn.MSELoss()
    optimizer_sgd = torch.optim.SGD(predict_net.parameters(), lr=1e-3)
    epoch = 0
    while True:
        print_loss = 100
        for step, (batch_x, batch_y) in enumerate(loader):  # 每一步loader释放一小批数据用来学习
            # 打印数据
            # print("epoch:", epoch, "step:", step, 'batch_x:', batch_x.numpy(), 'batch_y:', batch_y.numpy())
            print("epoch:", epoch)
            output = predict_net(batch_x)
            loss = criterion(output, batch_y)
            print(output)
            print(batch_y)
            print("...........")
            print_loss = loss.data
            optimizer_adam.zero_grad()
            loss.backward()
            optimizer_adam.step()
            print("均方误差:", np.mean(np.square(output.detach().numpy() - batch_y.detach().numpy())))
        epoch += 1
        print("loss", print_loss)
        if print_loss < 1e-3:
            print("==========End of Training==========")
            PATH = '../state_dict_model_good.pth'
            # 先建立路径
            torch.save(predict_net.state_dict(), PATH)
            # 理论值与预测值的均方误差
            break
