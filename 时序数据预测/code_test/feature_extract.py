import torch
import torch.nn as nn


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, input_size)
        output, (h_n, c_n) = self.lstm(x)
        # h_n 形状: (num_layers, batch_size, hidden_size)
        # 取最后一个时间步的隐藏状态作为特征表示
        feature_vector = h_n[-1]
        # 使用全连接层降维以获得最终的特征向量
        feature_vector = self.fc(feature_vector)
        return feature_vector

# 定义超参数
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 5
seq_len = 5
batch_size = 3

# 创建模型
model = LSTMFeatureExtractor(input_size, hidden_size, num_layers, output_size)
# 随机生成输入序列数据
inputs = torch.randn(batch_size, seq_len, input_size)
print("input_shape:",inputs.shape)
# 提取特征
features = model(inputs)
# 输出形状: (batch_size, output_size)
print(features.shape)
