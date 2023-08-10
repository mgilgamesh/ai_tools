import torch
import torch.nn as nn


# input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout

class MyTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super(MyTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                        dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(input_dim, output_dim)



    def generate_mask(self, size):
        # mask = (torch.tril(torch.ones(size, size)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = (torch.tril(torch.ones(size, size)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        print("mask1:", mask)
        mask_2 = nn.Transformer.generate_square_subsequent_mask(size,size)
        print("mask_2:", mask_2)
        return mask

    def forward(self, x):
        # 输入x的形状:(seq_len, batch_size, input_dim)
        mask = self.generate_mask(x.size(0)).to(x.device)
        output = self.transformer_encoder(x, mask)

        output = self.decoder(output)
        return output


if __name__ == "__main__":
    # 模型定义
    # input_dim = 10
    # output_dim = 5
    # hidden_dim = 32
    # num_layers = 2
    # num_heads = 2
    # dropout = 0.1

    input_dim = 1
    hidden_dim = 256
    num_layers = 2
    num_heads = 1
    output_dim = 1
    dropout = 0.1

    model = MyTransformer(input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout)
    # 输入数据
    seq_len = 10
    batch_size = 20

    x = torch.rand(seq_len, batch_size, 1)  # x的shape是(seq_len, batch_size, input_dim)
    # print("x:", x)
    # 通过模型计算输出
    output = model(x)
    # print(output)
