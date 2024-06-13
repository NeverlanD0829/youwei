from typing import Optional, Tuple, Union, Dict             # typing 模块用于类型提示
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchsummary import summary
from thop import profile


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:                         # d_model表示嵌入维度   max_len表示位置编码最大长度
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)                                          # 创建位置编码张量pe,初始全零化
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)         # 生成从0到 max_len-1的位置序列,为每个位置计算位置编码
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))    # div_term用于缩放位置
        pe[:, 0::2] = torch.sin(position * div_term)                                # 使用正余弦函数分别计算偶数和奇数位置的编码  
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                                        # 扩展一个维度
        self.register_buffer('pe', pe)                                              # 注册为模型缓冲区

    def forward(self, x: Tensor) -> Tensor:
        pe = self.pe.to(x.device)
        return x + pe[:, :x.size(1), :]                                             # 位置编码 pe 移动到输入张量 x 所在的设备，然后将其添加到输入张量中


class GlobalTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, ffn_dim: int, encoder_layers: int) -> None:
        super().__init__()
        self.d_model = d_model                                                      # d_model 是嵌入维度        n_head 是多头注意力的头数
        self.n_head = n_head                                                        # ffn_dim 是前馈网络的维度   encoder_layers 是编码器层数
        self.ffn_dim = ffn_dim
        self.encoder_layers = encoder_layers
        self.position_encoding = None  # Initialize later based on input shape
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(                                        # 初始化 Transformer 编码器层和编码器
            d_model=d_model, nhead=n_head, dim_feedforward=ffn_dim, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(                                                   # 初始化 Transformer 的编码器
            self.transformer_encoder_layer, num_layers=encoder_layers, norm=nn.LayerNorm(d_model)
        )

    def sequentialize(self, x: Tensor) -> Tuple[Tensor, dict]:                            # 将[N,C,H,W] 转换为 [N,H*W,C]
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(N, H * W, C)                                    # reshape 方法将其展平为序列
        shape_dict = {'origin_N': N, 'origin_C': C, 'origin_H': H, 'origin_W': W}
        return x, shape_dict

    def unsequentialize(self, x: Tensor, dim_dict: dict) -> Tensor:                       # 将序列化后的张量恢复为原始形状。view 方法根据形状字典重塑张量，permute 方法重新排列维度。
        x = x.contiguous().view(dim_dict['origin_N'], dim_dict['origin_H'], dim_dict['origin_W'], dim_dict['origin_C'])
        return x.permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        x, shape_dict = self.sequentialize(x)                                                   # 前向传播时调用。首先序列化输入张量，然后初始化位置编码（如果尚未初始化或大小不匹配）。
        if self.position_encoding is None or self.position_encoding.pe.size(1) != x.size(1):
            self.position_encoding = PositionalEncoding(d_model=self.d_model, max_len=x.size(1)).to(x.device)
        x = self.position_encoding(x)                                                           # 将位置编码添加到输入张量中
        x = self.transformer_encoder(x)
        x = self.unsequentialize(x, dim_dict=shape_dict)                                        # 最后将张量恢复为原始形状并返回
        return x


if __name__ == "__main__":
    inputs = torch.randn([1, 32, 44, 44]).to('cuda:0')
    d_model = 32
    n_head = 8
    ffn_dim = 48                        # TransformerEncoder中每个位置的前馈神经网络（Feed Forward Network）的维度
    encoder_layer = 3
    model = GlobalTransformerEncoder(d_model=d_model, n_head=n_head, ffn_dim=ffn_dim, encoder_layers=encoder_layer).to('cuda:0')
    out = model(inputs)
    print(out.shape)
    

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()}")  

    # Calculate FLOPs and parameters
    flops, params = profile(model, inputs=(inputs,))
    print(f"Total FLOPs: {flops}")
    print(f"Total parameters: {params}")