

import torch
import torch.nn as nn

# 设定输入和输出的维度
input_dim = 32
output_dim = 32
hidden_dim = 64

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)

    def forward(self, src):
        outputs, (hidden, cell) = self.lstm(src)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden, cell):
        outputs, (hidden, cell) = self.lstm(hidden, (hidden, cell))
        outputs = self.fc_out(outputs)
        return outputs


encoder = Encoder(input_dim, hidden_dim)
decoder = Decoder(hidden_dim, output_dim)

# 随机生成一些输入数据
# 假设我们有10个样本，每个样本长度为5，特征维度为input_dim
src = torch.randn(5, 10, input_dim)

# 通过Encoder
hidden, cell = encoder(src)

# 将Encoder的输出作为Decoder的输入
outputs = decoder(hidden, cell)

print(outputs)


