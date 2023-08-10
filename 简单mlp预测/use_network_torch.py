import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data

# 准备数据

if __name__ == "__main__":
    train_feature = np.zeros((10000, 2))
    train_label = np.zeros((10000, 1))
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.linspace(-np.pi, np.pi, 100)

    # 拟合出 -2 cosX + sinY


    for i in range(100):
        for j in range(100):
            train_feature[i * 100 + j][0] = x[i]
            train_feature[i * 100 + j][1] = y[j]
            train_label[i * 100 + j][0] = -2 * np.cos(x[i]) * np.sin(y[j])

    # 定义超参数
    lr = 1e-3
    K_epochs = 100

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    predict_net = nn.Sequential(
        nn.Linear(2, 128),
        nn.Tanh(),
        nn.Linear(128, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

    optimizer_adam = torch.optim.Adam([
        {'params': predict_net.parameters(), 'lr': lr}
    ])

    criterion = nn.MSELoss()
    optimizer_sgd = torch.optim.SGD(predict_net.parameters(), lr=1e-3)

    data_input = torch.from_numpy(train_feature).float()
    data_output = torch.from_numpy(train_label).float()

    torch_dataset = data.TensorDataset(data_input, data_output)

    # 把dataset放入DataLoader
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=1200,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）
        num_workers=2  # 多少线程来读取数据
    )

    epoch = 0
    while True:
        print_loss = 100
        for step, (batch_x, batch_y) in enumerate(loader):  # 每一步loader释放一小批数据用来学习
            # 打印数据
            # print("epoch:", epoch, "step:", step, 'batch_x:', batch_x.numpy(), 'batch_y:', batch_y.numpy())
            print("epoch:", epoch)
            output = predict_net(batch_x)
            loss = criterion(output, batch_y)
            print_loss = loss.data
            optimizer_adam.zero_grad()
            loss.backward()
            optimizer_adam.step()
            # if epoch % 50 == 0:
            #     print('Loss: {:.6f} after {} batches'.format(loss, epoch))
            #     print('==> Learned function:\t' + poly_desc(model.poly.weight.view(-1), model.poly.bias))
            print("均方误差:", np.mean(np.square(output.detach().numpy() - batch_y.detach().numpy())))
        epoch += 1
        print("loss", print_loss)
        if print_loss < 1e-4:
            print("==========End of Training==========")
            PATH = 'state_dict_model_good.pth'
            # 先建立路径
            torch.save(predict_net.state_dict(), PATH)
            # 理论值与预测值的均方误差
            break
