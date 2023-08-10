import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        #div_term = torch.exp(torch.arange(0, d_model, 3).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class MyTransformer_Pos(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super(MyTransformer_Pos, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                        dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(input_dim, output_dim)

    def generate_mask(self, size):
        mask = (torch.tril(torch.ones(size, size)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # print("mask:", mask)
        return mask

    def forward(self, x):
        # 输入x的形状:(seq_len, batch_size, input_dim)
        x = self.pos_encoder(x)  # Add positional encoding
        mask = self.generate_mask(x.size(0)).to(x.device)
        output = self.transformer_encoder(x, mask)
        output = self.decoder(output)
        return output


class MyTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super(MyTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                        dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 输入x的形状:(seq_len, batch_size, input_dim)
        output = self.transformer_encoder(x)
        output = self.decoder(output)
        return output


class All_Transformer(nn.Module):
    def __init__(self, input_dim_pos, output_dim_pos, input_dim, output_dim, hidden_dim, num_layers, num_heads,
                 dropout):
        super(All_Transformer, self).__init__()

        self.pos_encoder = PositionalEncoding(120)
        self.encoder_layer_pos = nn.TransformerEncoderLayer(d_model=input_dim_pos, nhead=num_heads,
                                                            dim_feedforward=hidden_dim,
                                                            dropout=dropout)
        self.liner_pos = nn.Linear(5,120)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                        dropout=dropout)
        self.liner = nn.Linear(2,120)

        self.transformer_encoder_pos = nn.TransformerEncoder(self.encoder_layer_pos, num_layers=num_layers)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_pos = nn.Linear(input_dim_pos, output_dim_pos)
        self.decoder = nn.Linear(input_dim, output_dim)
        self.out_layer_1 = nn.Linear(output_dim_pos + output_dim, 256)
        self.out_layer_2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()  # 添加 ReLU 激活函数

    def generate_mask(self, size):
        mask = (torch.tril(torch.ones(size, size)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # print("mask:", mask)
        return mask


    def forward(self, x):
        # 输入x的形状:(seq_len, batch_size, input_dim)
        # 将x进行变化
        x_left = x[:60, :, :1]
        # 获取最后一列的最后一个元素并复制60次
        last_val = x[54:55, :, -1:].repeat(60, 1, 1)

        x_left = torch.cat([x_left, last_val], dim=2)
        x_right = x[60:, :, 1:6]

        # print("x_left:", x_left.shape)
        # print("x_right:", x_right.shape)
        x_right = self.liner_pos(x_right)
        x_right = self.pos_encoder(x_right)
        mask = self.generate_mask(x_right.size(0)).to(x.device)

        # print("x_right",x_right.shape)
        # print("mask",mask.shape)

        output_x_right = self.transformer_encoder_pos(x_right,mask)

        x_left = self.liner(x_left)
        x_left = self.pos_encoder(x_left)
        
        output_x_left = self.transformer_encoder(x_left)
        # print("output_x_right:", output_x_right.shape)
        # print("output_x_left_right:", output_x_left.shape)
        output_x_right = self.decoder_pos(output_x_right)
        output_x_right = self.relu(output_x_right)
        output_x_left = self.decoder(output_x_left)
        output_x_left = self.relu(output_x_left)
        output_all = torch.cat((output_x_right, output_x_left), dim=2)
        # print("output_all:", output_all.shape)
        output_all = self.out_layer_1(output_all)
        output_all = self.out_layer_2(output_all)
        # print("output_all:", output_all.shape)
        # output_all = output_all.squeeze(-1)  # 去掉最后一维
        return output_all


if __name__ == "__main__":
    # 模型定义
    # input_dim = 10
    # output_dim = 5
    # hidden_dim = 32
    # num_layers = 2
    # num_heads = 2
    # dropout = 0.1

    input_dim_pos = 120
    hidden_dim = 256
    num_layers = 2
    num_heads = 20
    output_dim_pos = 120
    dropout = 0.1
    input_dim = 120
    output_dim = 120
    seq_len_pos = 120
    batch_size_pos = 20
    
    x_pos = torch.rand(seq_len_pos, batch_size_pos, 7)  # x的shape是(seq_len, batch_size, input_dim)
    print("x_pos:", x_pos.shape)
    all_trans_model = All_Transformer(input_dim_pos, output_dim_pos, input_dim, output_dim, hidden_dim, num_layers,
                                      num_heads, dropout)
    # print(all_trans_model)
    outdata = all_trans_model(x_pos)
    # print(outdata.shape)
    # input_dim_pos = 3
    # hidden_dim = 256
    # num_layers = 2
    # num_heads = 1
    # output_dim_pos = 3
    # dropout = 0.1
    # model_pos = MyTransformer_Pos(input_dim_pos, output_dim_pos, hidden_dim, num_layers, num_heads, dropout)
    #
    # input_dim = 2
    # output_dim = 2
    # model = MyTransformer(input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout)
    #
    # # 输入数据
    # seq_len_pos = 60
    # batch_size_pos = 20
    # x_pos = torch.rand(seq_len_pos, batch_size_pos, 3)  # x的shape是(seq_len, batch_size, input_dim)
    # seq_len = 60
    # batch_size = 20
    # x = torch.rand(seq_len_pos, batch_size, 2)
    # # print("x:", x)
    # # 通过模型计算输出
    # output_pos = model_pos(x_pos)
    # output = model(x)
    # print("output_pos:", output_pos.shape)
    # print("output:", output.shape)
