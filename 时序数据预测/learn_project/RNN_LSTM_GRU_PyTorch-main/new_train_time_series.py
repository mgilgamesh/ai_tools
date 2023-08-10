from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_sine_wave():
    np.random.seed(2)
    T = 20
    L = 1000
    N = 100

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')
    return data


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imp', '--implementation', type=str, default='custom', choices=['torch', 'custom'],
                        help='Implementation types to use')
    parser.add_argument('--arch', type=str, default='gru', choices=['rnn', 'lstm', 'gru'], help='LSTM types to use')
    parser.add_argument('--epochs', type=int, default=15, help='epochs to run')
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    print(args)

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = generate_sine_wave()
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])

    input_size = 1
    hidden_size = 51
    num_layers = 1
    output_size = 1

    net = LSTMFeatureExtractor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               output_size=output_size)
    net.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(net.parameters(), lr=0.8)

    input_data = input
    # begin to train
    for i in range(args.epochs):
        print('STEP: ', i)

        def closure():
            optimizer.zero_grad()
            input_data_new = input_data.unsqueeze(dim=2)
            # print("input_data_new.shape:", input_data_new.shape)
            out = net(input_data_new)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            test_input = test_input.unsqueeze(dim=2)
            pred = net(test_input)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()

        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)


        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)


        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('results/predict{}_{}_{}.png'.format(i, args.arch, args.imp))
        plt.close()
