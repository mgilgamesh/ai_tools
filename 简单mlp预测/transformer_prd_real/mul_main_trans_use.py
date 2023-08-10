

import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter
from model_use import MyTransformer

import matplotlib.pyplot as plt


if __name__ == '__main__':
    input_dim = 120
    hidden_dim = 256
    num_layers = 3
    num_heads = 5
    output_dim = 120
    dropout = 0.1

    # 读取全部数据
    pd_data = pd.read_csv("deal_data_0615_01_rename.csv")
    
    sigma1 = 49
    crise = pd_data["Crystal_rise"].values
    d = pd_data["diameter"].values
    d2 = pd_data["diameter"].diff().diff().values[2:]

    s_smooth_ori = scipy.ndimage.gaussian_filter1d(d2, sigma=sigma1)
    s_smooth = [(x - s_smooth_ori.min()) / (s_smooth_ori.max() - s_smooth_ori.min()) for x in s_smooth_ori]

    crises_smooth_ori = scipy.ndimage.gaussian_filter1d(crise, sigma=sigma1)
    crises_smooth = [(x - crises_smooth_ori.min()) / (crises_smooth_ori.max() - crises_smooth_ori.min()) for x in crises_smooth_ori]


    x_data = s_smooth[57:]
    y_data = crises_smooth[:]


    # plt.figure(figsize=(15, 3))
    # plt.plot(x_data, label="x_data")
    # plt.plot(y_data, label="y_data")

    # plt.legend()
    # plt.show()

    # corr, p_value = pearsonr(x_data[0:1000], y_data[0:1000])
    # print("corr:", corr, "p_value:", p_value)

    data_len = 120
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
    evl_data_len = 50
    train_data_len = len(all_data) - evl_data_len
    eval_data = all_data.sample(n=evl_data_len)
    train_data = all_data.sample(n=train_data_len)


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

    predict_net = MyTransformer(input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout)

    model_path = "model_pytorch/model_trans_end.pth"
    predict_net.load_state_dict(torch.load(model_path))

    eval_loader = data.DataLoader(
        dataset=eval_dataset,
        batch_size=1,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）
        num_workers=1  # 多少线程来读取数据
    )


    output_loss = 0
    predict_net.eval()  # 也可以不写，规范的话就写，用来表明是测试步骤
    eval_loss = 100
    real_output_data = []
    real_batch_y_data = []
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(eval_loader):
            # 这里的每一次循环 都是一个minibatch  一次for循环里面有64个数据。
            # batch_x_new = batch_x.permute(1, 0)
            batch_x_new = batch_x[np.newaxis,:, :]
            # print("batch_x_new.shape:", batch_x_new.shape)
            output = predict_net(batch_x_new)
            # print("output_shape:", output.shape)
            output_loss = output.squeeze()
            # print("output_loss:",output_loss.size())
            # print("batch_y:",batch_y.size())
            real_output_loss = output_loss * (max(s_smooth_ori) - min(s_smooth_ori)) + min(s_smooth_ori)
            real_batch_y = batch_y * (max(s_smooth_ori) - min(s_smooth_ori)) + min(s_smooth_ori)
            real_output_loss = real_output_loss.squeeze()
            real_batch_y = real_batch_y.squeeze()
            # print("real_output:",real_output_loss.size())
            # print("real_batch_y:",real_batch_y.size())

            real_output_data += real_output_loss.tolist()
            real_batch_y_data += real_batch_y.tolist()

    # print("real_output_data:",real_output_data)
    # print("real_batch_y_data:",real_batch_y_data)

    list_x =  list(range(len(real_output_data)))
    list_real_output_data = real_output_data
    list_real_batch_y_data = real_batch_y_data
    # 绘制两组数据
    plt.plot(list_real_output_data, label='prd_data')
    plt.plot(list_real_batch_y_data, label='real_data')
    
    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('Two Lists Comparison')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    # 显示图形
    plt.show()
    # print('测试集均方误差', np.mean(np.square(output_loss.detach().numpy() - batch_y.detach().numpy())))


















