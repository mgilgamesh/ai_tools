
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class MyTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super(MyTransformer, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout),
                                             num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        output = self.fc(self.encoder(x))
        return output

