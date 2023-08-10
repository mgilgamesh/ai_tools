import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch import nn

POLY_DEGREE = 3


def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE + 1)], 1)


W_target = torch.FloatTensor([3, 6, 2]).unsqueeze(1)
b_target = torch.FloatTensor([8])


def f(x):
    return x.mm(W_target) + b_target.item()


def get_batch(batch_size=64):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)  # + torch.rand(1)
    return Variable(x), Variable(y)


class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


model = poly_model()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def poly_desc(W, b):
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


epoch = 0
while True:
    batch_x, batch_y = get_batch()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.data
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # if epoch % 50 == 0:
    #     print('Loss: {:.6f} after {} batches'.format(loss, epoch))
    #     print('==> Learned function:\t' + poly_desc(model.poly.weight.view(-1), model.poly.bias))
    epoch += 1
    if print_loss < 1e-3:
        print()
        print("==========End of Training==========")
        break

print('Loss: {:.6f} after {} batches'.format(loss, epoch))
print('==> Learned function:\t' + poly_desc(model.poly.weight.view(-1), model.poly.bias))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
