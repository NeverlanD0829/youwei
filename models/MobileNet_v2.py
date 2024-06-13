import torch
from torch import nn
import torch.nn.functional as F
from thop import profile, clever_format
from torchsummary import summary



def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class B2_MobileNetV2(nn.Module):
    def __init__(self, num_classes=1, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(B2_MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(64 * width_mult, round_nearest)
        # last_channel = _make_divisible(1024 * width_mult, round_nearest)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # last_channel = _make_divisible(1280 * width_mult, round_nearest)

        # if inverted_residual_setting is None:           # t为Bottleneck内部升维的倍数，c为通道数，n为该bottleneck重复的次数，s为sride。
        #     inverted_residual_setting = [
        #         [1, 16, 1, 1],
        #         [6, 24, 2, 2],
        #         [6, 32, 3, 2],
        #         [6, 64, 4, 2],
        #         [6, 96, 3, 1],
        #         [6, 160, 3, 2],
        #         [6, 320, 1, 1],
        #     ]

        if inverted_residual_setting is None:           # t为Bottleneck内部升维的倍数，c为通道数，n为该bottleneck重复的次数，s为sride。
            inverted_residual_setting = [
                [1, 64, 1, 1],
                [4, 256, 3, 2],
                [4, 512, 3, 2],
                [2, 1024, 2, 2],
                [2, 2048, 1, 2],
                # [6, 1024, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty or a 4-element list, got {}".format(inverted_residual_setting))

        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier1 = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(last_channel, num_classes),
        # )
        # self.classifier2 = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(last_channel, num_classes),
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features[0](x)                 #  [1,64,176,176]
        x = self.features[1](x)                 #  [1,64,176,176]
        # x0 = self.maxpool(x)                    #  [1,64,88,88]
        x1 = self.features[2](x)                #  [1,256,88,88]

        x2 = self.features[3](x1)               #  
        x2 = self.features[4](x2)               #  [1,256,88,88]
        x2 = self.features[5](x2)               #  [1,512,44,44]
        x3 = self.features[6](x2)               #  [1,512,44,44]
        x4 = self.features[7](x3)               #  [1,512,44,44]  
        x5 = self.features[8](x4)               #  [1,1024,22,22]  
        x5 = self.features[9](x5)               #  [1,1024,22,22]  
        x6 = self.features[10](x5)              #  [1,2048,11,11] 
        return x,x1,x2,x3,x4,x5,x6
        



if __name__ == "__main__":
    images = torch.zeros(1, 3, 352, 352)
    model = B2_MobileNetV2()
    flops, params = profile(model, inputs=(images,))
    flops, params = clever_format([flops, params])
    print(f"FLOPS: {flops}, Params: {params}")

    if torch.cuda.is_available():
        model = model.to('cuda')
        images = images.to('cuda')
        summary(model, (3, 352, 352))


