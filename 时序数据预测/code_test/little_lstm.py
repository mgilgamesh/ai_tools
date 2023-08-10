import torch
import torch.nn as nn

input_size = 10
hidden_size = 20
num_layers = 2
seq_len = 5
batch_size = 3

# 创建一个LSTM层
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

# 随机生成一个输入序列
inputs = torch.randn(batch_size, seq_len, input_size)

# 初始化隐藏状态h0和细胞状态c0
h0 = torch.randn(num_layers, batch_size, hidden_size)
c0 = torch.randn(num_layers, batch_size, hidden_size)

# 将输入数据、初始隐藏状态和初始细胞状态传递给LSTM
output, (h_n, c_n) = lstm(inputs, (h0, c0))
dd = "asd"
