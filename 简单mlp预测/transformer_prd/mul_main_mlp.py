# 找到（频率，流量）对应的功率

import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.stats import pearsonr
from model_use import MyTransformer
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    input_dim = 1
    hidden_dim = 256
    num_layers = 2
    num_heads = 1
    output_dim = 1
    dropout = 0.1

    # 读取全部数据
    pd_data = pd.read_csv("deal_data_0615_01_rename.csv")
    writer = SummaryWriter('runs/training_logs_mlp')
    sigma1 = 49

    crise = pd_data["Crystal_rise"].values
    d = pd_data["diameter"].values
    d2 = pd_data["diameter"].diff().diff().values[2:]

    s_smooth = scipy.ndimage.gaussian_filter1d(d2, sigma=sigma1)
    s_smooth = [(x - s_smooth.mean()) / s_smooth.std() for x in s_smooth]

    crises_smooth = scipy.ndimage.gaussian_filter1d(crise, sigma=sigma1)
    crises_smooth = [(x - crises_smooth.mean()) / crises_smooth.std() for x in crises_smooth]

    x_data = s_smooth[57:]
    y_data = crises_smooth[:]

    plt.figure(figsize=(15, 3))
    plt.plot(x_data, label="x_data")
    plt.plot(y_data, label="y_data")

    plt.legend()
    plt.show()

    corr, p_value = pearsonr(x_data[0:1000], y_data[0:1000])
    print("corr:", corr, "p_value:", p_value)

    data_len = 10
    x_data_new = np.array(x_data[0:39840]).reshape(-1, data_len)
    y_data_new = np.array(y_data[0:39840]).reshape(-1, data_len)
    all_data_new = np.concatenate((x_data_new, y_data_new), axis=1)
    print("all_data_new_shape:", all_data_new.shape)
    all_data = pd.DataFrame(data=all_data_new)
    # print("x_data:",len(x_data))
    # print("y_data:",len(y_data))

    # all_data = pd.DataFrame({'d2_smooth': x_data[0:39841], 'crises_smooth': y_data[0:39841]})
    print('Starting training')
    # 数据分离
    evl_data_len = 3000
    train_data_len = len(all_data) - evl_data_len
    eval_data = all_data.sample(n=evl_data_len)
    train_data = all_data.sample(n=train_data_len)
    '''
    # 归一化模型存储
    all_data_output = all_data[['Crystal_rise']]
    all_data_output = np.array(all_data_output)
    scaler = StandardScaler()
    scaler.fit(all_data_output)
    joblib.dump(scaler, 'scaler.pkl')
    '''

    # 训练数据集的输入和输出
    train_input = train_data.iloc[:, :data_len]
    train_output = train_data.iloc[:, data_len:]

    # 评估数据集的输入和输出
    eval_input = eval_data.iloc[:, :data_len]
    eval_output = eval_data.iloc[:, data_len:]

    # 转 torch数据集
    train_input = torch.from_numpy(np.array(train_input)).float()
    print("train_input_shape:", train_input.shape)
    train_output = torch.from_numpy(np.array(train_output)).float()
    train_dataset = data.TensorDataset(train_input, train_output)

    # 转 torch数据集
    eval_input = torch.from_numpy(np.array(eval_input)).float()
    eval_output = torch.from_numpy(np.array(eval_output)).float()
    eval_dataset = data.TensorDataset(eval_input, eval_output)

    # 构建数据集的loader
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=100,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）
        num_workers=1  # 多少线程来读取数据
    )
    eval_loader = data.DataLoader(
        dataset=eval_dataset,
        batch_size=100,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）
        num_workers=1  # 多少线程来读取数据
    )
    # 机器学习超参数及神经网络模型
    lr = 1e-3
    K_epochs = 100

    predict_net = nn.Sequential(
        nn.BatchNorm1d(data_len),
        nn.Linear(data_len, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, data_len),
    )
    # 定义优化器
    optimizer_adam = torch.optim.Adam([
        {'params': predict_net.parameters(), 'lr': lr}
    ])
    criterion = nn.MSELoss()
    optimizer_sgd = torch.optim.SGD(predict_net.parameters(), lr=1e-3)
    epoch = 0
    while True:
        print_loss = 100
        print("epoch:", epoch)
        running_loss = 0
        i = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):  # 每一步loader释放一小批数据用来学习
            # 打印数据
            # print("epoch:", epoch, "step:", step, 'batch_x:', batch_x.numpy(), 'batch_y:', batch_y.numpy())
            output = predict_net(batch_x)
            # print("output_shape:", output.shape)
            loss = criterion(output, batch_y)
            # print("...........")
            print_loss = loss.data
            optimizer_adam.zero_grad()
            loss.backward()
            optimizer_adam.step()

            i += 1
            # 打印统计信息
            running_loss += loss.item()
            if i % 100 == 99:  # 每100个批次打印一次
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                # 写入TensorBoard
                writer.add_scalar('training loss', running_loss / 100,
                                  epoch * len(train_loader) + i)
                running_loss = 0.0
            # print("均方误差:", np.mean(np.square(output_loss.detach().numpy() - batch_y.detach().numpy())))
        print("均方误差:", np.mean(np.square(output.detach().numpy() - batch_y.detach().numpy())))
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
                # print("batch_x_new.shape:", batch_x_new.shape)
                output = predict_net(batch_x)
                # print("output_shape:", output.shape)
                loss_in = criterion(output, batch_y)
                sum_loss += loss_in
                eval_loss = loss_in

        # print('第{}轮测试集的正确率:{:.2f}%'.format(epoch + 1, accurate / len(eval_dataset) * 100))
        print('测试集均方误差', np.mean(np.square(output.detach().numpy() - batch_y.detach().numpy())))
        # print("eval_loss:",eval_loss)
        if epoch >= 400 or eval_loss < 1e-3:
            torch.save(predict_net, 'model_pytorch/model_mlp_{}.pth'.format(epoch + 1))
        if eval_loss < 1e-4 and epoch >= 2000:
            break
