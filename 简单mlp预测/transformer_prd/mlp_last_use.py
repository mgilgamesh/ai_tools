import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, mask):
        x = self.fc1(x)
        x = self.relu(x)
        masked_output = torch.mul(x, mask)  # 将输出与mask相乘
        output = self.fc2(masked_output)
        return output


# 定义模型参数
input_size = 10
hidden_size = 10
output_size = 1
# 创建MLP模型实例
model = MLP(input_size, hidden_size, output_size)
# 创建输入数据
input_data = torch.randn((5, input_size))
mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 设计一个只与过去的输入数据相关的mask
                     [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
# 前向传播计算输出
output = model(input_data, mask)
print(output)
