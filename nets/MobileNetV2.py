import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


# SE注意力模块
class SeModule(nn.Module):
    # ratio代表第一个全连接下降通道的倍数
    def __init__(self, in_channel, ratio=4):
        super().__init__()
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 第一个全连接层将特征图的通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # relu激活，可自行换别的激活函数
        self.relu = nn.ReLU()
        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):  # inputs 代表输入特征图
        b, c, h, w = inputs.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b, c])
        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)
        x = self.relu(x)
        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)
        # 对通道权重归一化处理
        x = self.sigmoid(x)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])
        # 将输入特征图和通道权重相乘
        outputs = x * inputs
        return outputs


# _make_divisible确保Channel个数能被8整除，很多嵌入式设备做优化时都采用这个准则
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    # 对变量 v 进行舍入到最接近的 divisor 的整数倍的操作，并确保结果不小于 min_value
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# 最后的groups参数：groups=1时，普通卷积；groups=输入通道数in_planes时，DW卷积=深度可分离卷积
# pytorch官方继承自nn.sequential，想用它的预训练权重，就得听它的

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


# 参数顺序：输入通道数,输出通道数，步长，变胖倍数(扩展因子)
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]                                     # 断言条件是否为真,s不在1,2之间时,触发错误

        hidden_dim = int(round(inp * expand_ratio))                 # 隐藏维度,输入通道数*t,将输入变量 inp 乘以t,四舍五入到最接近的整数
        self.use_res_connect = self.stride == 1 and inp == oup      # 只有同时满足两个条件时，才使用短连接

        layers = []

        if expand_ratio != 1:                                       # 如果扩展因子等于1，就没有第一个1x1的卷积层
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # 3x3 depthwise conv，因为使用了groups=hidden_dim
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)                          # nn.Sequential 创建了一个神经网络模块，其中包含了一系列的层（layers）

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# MobileNetV2是一个类，继承自nn.module这个父类
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): 输出类别数目
            width_mult (float): 调整每个层中通道的数量的比例
            inverted_residual_setting: 网络结构
            round_nearest (int): 将每个层中的通道数四舍五入为这个数字的倍数。
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        # 保证通道数是 8 的倍数，原因是：适配于硬件优化加速
        input_channel = _make_divisible(32 * width_mult, round_nearest)
        last_channel = _make_divisible(1280 * width_mult, round_nearest)

        if inverted_residual_setting is None:
            # t表示扩展因子(变胖倍数); c是通道数; n是block重复几次;
            # s：stride步长,只针对第一层,其它s都等于1
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # 检查t, c, n, s是否都在
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty ""or a 4-element list, got {}".format(inverted_residual_setting))

        # conv1 layer
        features = [ConvBNReLU(1, input_channel, stride=2)]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):         # s为1或者2 只针对重复了n次的bottleneck 的第一个bottleneck,重复n次的剩下几个bottleneck中s均为1
                stride = s if i == 0 else 1
                # 这个block就是上面那个InvertedResidual函数
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                # 这一层的输出通道数作为下一层的输入通道数
                input_channel = output_channel

        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        # *features表示位置信息，将特征层利用nn.Sequential打包成一个整体
        self.features = nn.Sequential(*features)

        # building classifier
        # 自适应平均池化下采样层，输出矩阵高和宽均为1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )
        print(self.classifier)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')           # 针对ReLU激活函数的0均值正态分布初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):                             # 权重初始化为1，偏置初始化为0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):                                  # 均值为0、标准差为0.01的正态分布初始化权重，偏置初始化为零
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平处理
        x = self.classifier(x)
        return x


# if __name__=="__main__":
#     model = MobileNetV2(num_classes=1, width_mult=1.0)
#     input_tensor = torch.randn(4, 224, 224).unsqueeze(1)
#     output_tensor = model(input_tensor)
#     print("Output Shape:", output_tensor.shape)
#     print("Output Values:", output_tensor)