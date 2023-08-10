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
    all_data = pd.read_csv("train_deal_data.csv")
    # 数据分配，训练集和测试集
    evl_data_len = 250
    train_data_len = len(all_data) - evl_data_len
    eval_data = all_data.sample(n=evl_data_len)
    train_data = all_data.sample(n=train_data_len)

    # 定义训练数据集的输入和输出
    train_input = train_data.iloc[:, :16]
    train_output = train_data.iloc[:, -2:]

    # 评估数据集的输入和输出
    eval_input = eval_data.iloc[:, :16]
    eval_output = eval_data.iloc[:, -2:]

    # 训练数据转 torch数据集
    train_input = torch.from_numpy(np.array(train_input)).float()
    train_output = torch.from_numpy(np.array(train_output)).float()
    train_dataset = data.TensorDataset(train_input, train_output)

    # 测试数据转 torch数据集
    eval_input = torch.from_numpy(np.array(eval_input)).float()
    eval_output = torch.from_numpy(np.array(eval_output)).float()
    eval_dataset = data.TensorDataset(eval_input, eval_output)

    # 构建训练数据集的loader
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=100,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）
        num_workers=1  # 多少线程来读取数据
    )
    # 构建测试数据集的loader
    eval_loader = data.DataLoader(
        dataset=eval_dataset,
        batch_size=100,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）
        num_workers=1  # 多少线程来读取数据
    )

    # 机器学习超参数及神经网络模型
    lr = 1e-3
    K_epochs = 100
    # 构建神经网络模型架构
    predict_net = nn.Sequential(
        nn.BatchNorm1d(16),
        nn.Linear(16, 128),
        nn.ReLU(),
        nn.Linear(128, 512),
        nn.ReLU(),
        nn.Linear(512, 2),
    )

    # 定义优化器和损失函数
    optimizer_adam = torch.optim.Adam([
        {'params': predict_net.parameters(), 'lr': lr}
    ])
    criterion = nn.MSELoss()
    optimizer_sgd = torch.optim.SGD(predict_net.parameters(), lr=1e-3)

    epoch = 0

    while True:
        print_loss = 100
        print("epoch:", epoch)
        for step, (batch_x, batch_y) in enumerate(train_loader):  # 每一步loader释放一小批数据用来学习
            # 打印数据
            # print("epoch:", epoch, "step:", step, 'batch_x:', batch_x.numpy(), 'batch_y:', batch_y.numpy())
            output = predict_net(batch_x)
            loss = criterion(output, batch_y)
            print_loss = loss.data
            optimizer_adam.zero_grad()
            loss.backward()
            optimizer_adam.step()
            print("均方误差:", np.mean(np.square(output.detach().numpy() - batch_y.detach().numpy())))
        print("print_loss:", print_loss)
        epoch += 1
        # 完成一个epoch 执行模型评估
        sum_loss = 0  # 记录总体损失值
        # 每轮训练完成跑一下测试数据看看情况
        accurate = 0
        predict_net.eval()  # 也可以不写，规范的话就写，用来表明是测试步骤
        eval_loss = 100
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(eval_loader):
                # 这里的每一次循环 都是一个minibatch  一次for循环里面有64个数据。
                output = predict_net(batch_x)
                loss_in = criterion(output, batch_y)
                sum_loss += loss_in
                eval_loss = loss_in
                # 计算在axis为1的轴上数据相等的个数
                equal_elements = torch.eq(output, batch_y)
                equal_counts = equal_elements.sum(dim=1)
                # 统计相等元素的个数
                num_equal_rows = torch.sum(equal_counts == 2).item()
                accurate += num_equal_rows
        print('第{}轮测试集的正确率:{:.2f}%'.format(epoch + 1, accurate / len(eval_dataset) * 100))
        print('测试集损失', sum_loss)
        print("eval_loss:", eval_loss)
        if epoch >= 400 or eval_loss < 1e-3:
            torch.save(predict_net.state_dict(), 'model_pytorch/model_{}.pth'.format(epoch + 1))
        if eval_loss < 1e-4:
            PATH = 'model_pytorch/state_dict_model_good.pth'
            # 先建立路径
            torch.save(predict_net.state_dict(), PATH)
            # 理论值与预测值的均方误差
            break

    pass
