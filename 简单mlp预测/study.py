import numpy as np
import matplotlib.pyplot as plt
import torch

# x = np.linspace(0,1000,20)
# y = np.linspace(0,500,20)
#
#
# X,Y = np.meshgrid(x, y)
#
# plt.plot(X, Y,
#          color='limegreen',  # 设置颜色为limegreen
#          marker='.',  # 设置点类型为圆点
#          linestyle='')  # 设置线型为空，也即没有线连接点
# plt.grid(True)
# plt.show()


POLY_DEGREE = 3


def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE + 1)], 1)


batch_size = 12

dd = make_features(torch.randn(12))

print(dd)
W_target = torch.FloatTensor([3, 6, 2]).unsqueeze(1)
b_target = torch.FloatTensor([8])


def f(x):
    return x.mm(W_target) + b_target.item()

ss = f(dd)
print(ss)

