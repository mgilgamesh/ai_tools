



import torch
import torch.utils.data as data

# 创建数据
x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# 先转换成torch能识别的dataset
torch_dataset = data.TensorDataset(x, y)

# 把dataset放入DataLoader
loader = data.DataLoader(
    dataset=torch_dataset,
    batch_size=4,             # 每批提取的数量
    shuffle=True,             # 要不要打乱数据（打乱比较好）
    num_workers=2             # 多少线程来读取数据
)

if __name__ == '__main__':
    for epoch in range(3):    # 对整套数据训练3次
        for step, (batch_x, batch_y) in enumerate(loader):  # 每一步loader释放一小批数据用来学习
            # 训练过程
            print("epoch:", epoch, "step:", step, 'batch_x:', batch_x, 'batch_y:', batch_y)
            # 打印数据
            print("epoch:", epoch, "step:", step, 'batch_x:', batch_x.numpy(), 'batch_y:', batch_y.numpy())

            nn = 4













