import torch
import torch.nn as nn
import numpy as np
# 
class Encoder(nn.Module):
     def __init__(self, input_size, hidden_size, output_size):
         super(Encoder, self).__init__()
         self.fc1 = nn.Linear(input_size, hidden_size)
         self.fc2 = nn.Linear(hidden_size, output_size)
     def forward(self, x):
         x = torch.relu(self.fc1(x))
         x = self.fc2(x)
         return x
#  
class Decoder(nn.Module):
     def __init__(self, input_size, hidden_size, output_size):
         super(Decoder, self).__init__()
         self.fc1 = nn.Linear(input_size, hidden_size)
         self.fc2 = nn.Linear(hidden_size, output_size)
     def forward(self, x):
         x = torch.relu(self.fc1(x))
         x = self.fc2(x)
         return x

class Autoencoder(nn.Module):
     def __init__(self, input_size, hidden_size, output_size):
         super(Autoencoder, self).__init__()
         self.encoder = Encoder(input_size, hidden_size, output_size)
         self.decoder = Decoder(output_size, hidden_size, input_size)
     def forward(self, x):
         x = self.encoder(x)
         x = self.decoder(x)
         return x
 # 测试数据
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

# # 转换为NumPy数组
# data_array = np.array(data)
# # 计算每列的最小值和最大值
# min_vals = np.min(data_array)
# max_vals = np.max(data_array)

# # 最小-最大归一化
# normalized_data = (data_array - min_vals) / (max_vals - min_vals)

# print("normalized_data:",normalized_data)

# data = normalized_data

 # 将数据转换为张量
data = torch.tensor(data).float()
 # 定义autoencoder
input_size = 3
hidden_size = 60
output_size = 2
autoencoder = Autoencoder(input_size, hidden_size, output_size)
 # 定义优化器和损失函数
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
 # 训练autoencoder
num_epochs = 1000
epoch = 0
while epoch <= 20000:
    # 向前传递
    outputs = autoencoder(data)
    loss = criterion(outputs, data)
     # 反向传递和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
     # 输出损失值
    if (epoch) % 100 == 0:
        # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, loss.item()))
        pass
    
 # 将列表编码为较低维度的向量
encoded_data = autoencoder.encoder(data)
print("Encoded data:", encoded_data)
 # 将向量解码回原始列表
decoded_data = autoencoder.decoder(encoded_data)
print("Decoded data:", decoded_data)