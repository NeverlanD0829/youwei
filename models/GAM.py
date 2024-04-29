"""
GAM 注意力机制：对CBAM注意力进行改进
先通道注意力，再空间注意力
"""
import torch
import torch.nn as nn


# 通道注意力
class Channel_Attention(nn.Module):
    def __init__(self, in_channel, out_channel, ratio=4):
        super(Channel_Attention, self).__init__()
        self.fc1 = nn.Linear(in_channel, in_channel // ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channel // ratio, in_channel)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # b, c, h, w = x.size()
        input = x.permute(0, 3, 2, 1)
        output = self.fc2(self.relu(self.fc1(input)))
        output = output.permute(0, 3, 2, 1)
        return output * x


# 空间注意力
class Spatial(nn.Module):
    def __init__(self, in_channel, out_channel, ratio, kernel_size=7):
        super(Spatial, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channel, in_channel // ratio, kernel_size=7, padding=padding
        )
        self.bn = nn.BatchNorm2d(in_channel // ratio)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channel // ratio, in_channel, kernel_size=kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        conv1 = self.act(self.bn(self.conv1(x)))
        conv2 = self.bn1(self.conv2(conv1))
        output = self.sig(conv2)
        return x * output

class GAM(nn.Module):
    def __init__(self,in_channel, out_channel, ratio = 4, kernel_size = 7):
        super(GAM, self).__init__()
        self.channel_attention = Channel_Attention(in_channel,out_channel,ratio)
        self.spatial_attention = Spatial(in_channel,out_channel,ratio,kernel_size)

    def forward(self, x):
        input = self.channel_attention(x)
        output= self.spatial_attention(input)
        return output

if __name__ == '__main__':
    input = torch.randn(1, 32, 128, 128).cuda()
    model = GAM(32, 32).cuda()
    output = model(input)
    print(output.size())

