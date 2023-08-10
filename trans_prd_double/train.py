import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter
from model_use import All_Transformer
from torch.utils.data import Dataset, TensorDataset
from sklearn.preprocessing import MinMaxScaler


class MyDataset(Dataset):
    def __init__(self, data, seq_length=120):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        # 减去seq_length以避免越界，也就是说我们将使用前 len(self.data) - self.seq_length + 1 个序列作为数据集
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, index):
        # 对于每一个索引，我们都返回一个连续的序列
        seq = self.data[index: index + self.seq_length]
        # 将序列分割为输入和输出
        input_data = torch.cat([seq[:60, :2], seq[60:, 2:]], dim=1)
        output_data = seq[60:, :2]
        return input_data, output_data


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("have a gpu")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    # pd_data = pd.read_csv("si_data_Time_2.csv")
    # scaler = MinMaxScaler()
    # # 创建一个新的DataFrame，包含原DataFrame的前两列
    # # 归一化的数据
    # df_norm = pd.DataFrame()
    # # 对第二列之后的每一列进行归一化，并将结果放在新的DataFrame里
    # for col in pd_data.columns[2:]:
    #     df_norm[col] = scaler.fit_transform(pd_data[[col]])[:, 0]
    # # print(df_new)
    # df_norm.to_csv("data.csv", index=False)
    df_norm = pd.read_csv("/home/mz/software/revive_new/revive_new/trans_prd_double/data.csv")
    # evl_data_len = 4000
    # train_data_len = len(pd_data) - evl_data_len
    # eval_data = pd_data.sample(n=evl_data_len)
    # train_data = pd_data.sample(n=train_data_len)

    evl_data_start = 20000
    eval_data_end = 24000

    # train_data_len = len(df_norm) - evl_data_len
    train_data = df_norm.iloc[0:evl_data_start].append(df_norm.iloc[eval_data_end:])
    eval_data = df_norm.iloc[evl_data_start:eval_data_end]
    

    # 初始化输入数据和目标数据的列表
    input_data_train = []
    target_data_train = []
    # 以120为步长，遍历整个DataFrame
    for i in range(0, train_data.shape[0] - 120 + 1):
        # 取出当前的120行数据
        sequence = train_data.iloc[i: i + 120]
        # 前60行的前2列是输入数据
        # input_sequence = sequence.iloc[:60, []].values
        first_col = sequence.iloc[:60, 0]
        # 选取最后一列的最后一个数据，并复制60次
        last_val = pd.Series([sequence.iloc[54, -1]] * 60)
        # 合并这两列数据，生成新的DataFrame
        # new_data = pd.concat([first_col, last_val], axis=1)
        input_sequence = sequence.iloc[:, :].values
        # 后60行的前2列是目标数据
        target_sequence = sequence.iloc[60:, -1:].values
        input_data_train.append(input_sequence)
        target_data_train.append(target_sequence)

    input_data_eval = []
    target_data_eval = []
    # 以120为步长，遍历整个DataFrame
    for i in range(0, eval_data.shape[0] - 120 + 1):
        # 取出当前的120行数据
        sequence = eval_data.iloc[i: i + 120]
        # 前60行的前2列是输入数据
        # input_sequence = sequence.iloc[:60, []].values
        first_col = sequence.iloc[:60, 0]
        # 选取最后一列的最后一个数据，并复制60次
        last_val = pd.Series([sequence.iloc[54, -1]] * 60)
        # 合并这两列数据，生成新的DataFrame
        # new_data = pd.concat([first_col, last_val], axis=1)
        input_sequence = sequence.iloc[:, :].values
        # 后60行的前2列是目标数据
        target_sequence = sequence.iloc[60:, -1:].values
        input_data_eval.append(input_sequence)
        target_data_eval.append(target_sequence)

    # 将数据列表转换为tensor
    input_data_train = torch.tensor(input_data_train, dtype=torch.float)
    target_data_train = torch.tensor(target_data_train, dtype=torch.float)
    # 构建TensorDataset
    dataset_train = TensorDataset(input_data_train, target_data_train)

    # 将数据列表转换为tensor
    input_data_eval = torch.tensor(input_data_eval, dtype=torch.float)
    target_data_eval = torch.tensor(target_data_eval, dtype=torch.float)
    # 构建TensorDataset
    dataset_eval = TensorDataset(input_data_eval, target_data_eval)
    print("get_data")
    # 构建数据集的loader
    train_loader = data.DataLoader(
        dataset=dataset_train,
        batch_size=80,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）
        num_workers=1  # 多少线程来读取数据
    )

    eval_loader = data.DataLoader(
        dataset=dataset_eval,
        batch_size=80,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）
        num_workers=1  # 多少线程来读取数据
    )

    lr = 1e-4
    K_epochs = 100

    input_dim_pos = 120
    hidden_dim = 256
    num_layers = 2
    num_heads = 12
    output_dim_pos = 120
    dropout = 0.4
    input_dim = 120
    output_dim = 120
    seq_len_pos = 120
    batch_size_pos = 20

    writer = SummaryWriter('training_logs_trans')
    predict_net = All_Transformer(input_dim_pos, output_dim_pos, input_dim, output_dim, hidden_dim, num_layers,
                                  num_heads, dropout)
    predict_net.load_state_dict(torch.load("/home/mz/software/revive_new/revive_new/trans_prd_double/model_pytorch/model_trans_101.pth"))
    predict_net.to(device)
    # 定义优化器
    optimizer_adam = torch.optim.Adam([
        {'params': predict_net.parameters(), 'lr': lr}
    ])
    criterion = nn.MSELoss()
    epoch = 0
    while True:
        print_loss = 100
        print("epoch:", epoch)
        running_loss = 0
        i = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):  # 每一步loader释放一小批数据用来学习
            # 打印数据
            # print("epoch:", epoch, "step:", step, 'batch_x:', batch_x.numpy(), 'batch_y:', batch_y.numpy())
            # print("batch_x_shape:", batch_x.shape)
            batch_x_new = batch_x.permute(1, 0, 2).to(device)
            output = predict_net(batch_x_new)
            # print("output:", output.shape)
            output_loss = output.permute(1, 0, 2)
            output_loss = output_loss.squeeze().to(device)
            # print("batch_y_shape:", batch_y.shape)
            batch_y_new = batch_y.squeeze().to(device)
            # print("output_loss_shape:", output_loss.shape)
            loss = criterion(output_loss, batch_y_new)
            print_loss = loss.data
            optimizer_adam.zero_grad()
            loss.backward()
            optimizer_adam.step()
            i += 1
            running_loss += loss.item()
            if i % 10 == 9:  # 每100个批次打印一次
                # 写入TensorBoard
                writer.add_scalar('training_loss', running_loss / 10)
                running_loss = 0.0

        print("均方误差:", np.mean(np.square(output_loss.cpu().detach().numpy() - batch_y_new.cpu().detach().numpy())))
        epoch += 1

        sum_loss = 0  # 记录总体损失值
        # 每轮训练完成跑一下测试数据看看情况
        accurate = 0
        predict_net.eval()  # 也可以不写，规范的话就写，用来表明是测试步骤
        eval_loss = 100
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(eval_loader):
                # 这里的每一次循环 都是一个minibatch  一次for循环里面有64个数据。
                # batch_x_new = batch_x.permute(1, 0, 2)
                # output = predict_net(batch_x_new)
                # # print("output:", output.shape)
                # output_loss = output.permute(1, 0, 2)
                # output_loss = output_loss.squeeze()
                # # print("batch_y_shape:", batch_y.shape)
                # batch_y_new = batch_y.squeeze()
                # # print("output_loss_shape:", output_loss.shape)
                # loss = criterion(output_loss, batch_y_new)
                batch_x_new = batch_x.permute(1, 0, 2).to(device)
                output = predict_net(batch_x_new)
                # print("output:", output.shape)
                output_loss = output.permute(1, 0, 2)
                output_loss = output_loss.squeeze().to(device)
                # print("batch_y_shape:", batch_y.shape)
                batch_y_new = batch_y.squeeze().to(device)
                # print("output_loss_shape:", output_loss.shape)
                loss = criterion(output_loss, batch_y_new)


        # print('第{}轮测试集的正确率:{:.2f}%'.format(epoch + 1, accurate / len(eval_dataset) * 100))
        print('测试集均方误差', np.mean(np.square(output_loss.cpu().detach().numpy() - batch_y_new.cpu().detach().numpy())))
        # print("eval_loss:",eval_loss)
        if epoch % 100 == 0 or eval_loss < 1e-3:
            torch.save(predict_net.state_dict(), 'model_pytorch/model_trans_{}.pth'.format(epoch + 1))
        if eval_loss < 1e-4 or epoch >= 1500:
            writer.close()
            torch.save(predict_net.state_dict(), 'model_pytorch/model_trans_end.pth'.format(epoch + 1))
            break


