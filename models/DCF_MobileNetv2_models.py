import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile,clever_format
from torchsummary import summary

from models.HolisticAttention import HA
# from model.ResNet import B2_MobileNetV2


# from HolisticAttention import HA
# from ResNet import B2_MobileNetV2
from models.MobileNet_v2 import B2_MobileNetV2

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

class DCF_MobileNetV2(nn.Module):
    def __init__(self, channel=32):
        super(DCF_MobileNetV2, self).__init__()
        self.mobilenet_v2 = B2_MobileNetV2()
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = aggregation(channel)

        self.rfb2_2 = RFB(512, channel)
        self.rfb3_2 = RFB(1024, channel)
        self.rfb4_2 = RFB(2048, channel)
        self.agg2 = aggregation(channel)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.HA = HA()
        if self.training:
            self.initialize_weights()

    def forward(self, x):

        x = self.mobilenet_v2.features[0](x)                 #  [1,64,176,176]
        x = self.mobilenet_v2.features[1](x)                 #  [1,64,176,176]
        # x0 = self.maxpool(x)                               #  [1,64,88,88]
        x1 = self.mobilenet_v2.features[2](x)                #  [1,256,88,88]

        x2 = self.mobilenet_v2.features[3](x1)               #  
        x2 = self.mobilenet_v2.features[4](x2)               #  [1,256,88,88]
        x2 = self.mobilenet_v2.features[5](x2)               #  [1,512,44,44]
        x3 = self.mobilenet_v2.features[6](x2)               #  [1,512,44,44]
        x4 = self.mobilenet_v2.features[7](x3)               #  [1,512,44,44]  
        x5 = self.mobilenet_v2.features[8](x4)               #  [1,1024,22,22]  
        x5 = self.mobilenet_v2.features[9](x5)               #  [1,1024,22,22]  
        x6 = self.mobilenet_v2.features[10](x5)              #  [1,2048,11,11] 

        x2_1 = x4                                            # [1,512,44,44]  
        x3_1 = x5                                            # [1,1024,22,22]  
        x4_1 = x6                                            # [1,2048,11,11] 
        x2_1 = self.rfb2_1(x2_1)                             # [1,32,44,44]       
        x3_1 = self.rfb3_1(x3_1)                             # [1,32,22,22]
        x4_1 = self.rfb4_1(x4_1)                             # [1,32,11,11]            
        attention_map = self.agg1(x4_1, x3_1, x2_1)          # [1,1,44,44]


        x2_2 = self.HA(attention_map.sigmoid(), x4)          # [1,512,44,44]
        x3_2 = x5                                            # [1,1024,22,22]
        x4_2 = x6                                            # [1,2048,11,11]
        x2_2 = self.rfb2_2(x2_2)                             # [1,32,44,44]  
        x3_2 = self.rfb3_2(x3_2)                             # [1,32,22,22]
        x4_2 = self.rfb4_2(x4_2)                             # [1,32,11,11] 
        detection_map = self.agg2(x4_2, x3_2, x2_2)          # [1,1,44,44]


        return self.upsample(attention_map), self.upsample(detection_map), x2_2, x3_2, x4_2

    def initialize_weights(self):
        mobilenet_v2 = B2_MobileNetV2()
        pretrained_dict = mobilenet_v2.state_dict()
        model_dict = self.mobilenet_v2.state_dict()

        all_params = {}
        for k, v in model_dict.items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif 'classifier1.1' in k:
                name = k.replace('classifier1.1', 'classifier1')  # 调整 classifier1 的命名
                if name in pretrained_dict.keys():
                    v = pretrained_dict[name]
                    all_params[k] = v
            elif 'classifier2.1' in k:
                name = k.replace('classifier2.1', 'classifier2')  # 调整 classifier2 的命名
                if name in pretrained_dict.keys():
                    v = pretrained_dict[name]
                    all_params[k] = v

        assert len(all_params.keys()) == len(model_dict.keys()), f"不匹配: {len(all_params.keys())} vs {len(model_dict.keys())}"
        self.mobilenet_v2.load_state_dict(all_params)




if __name__=="__main__":
    images = torch.zeros(1,3,352,352)
    model_rgb = DCF_MobileNetV2()
    result = model_rgb(images)
    print(result)
    flops, params = profile(model_rgb, inputs=(images,))
    flops, params = clever_format([flops, params])
    print(f"FLOPS: {flops}, Params: {params}")

    if torch.cuda.is_available():
        model_rgb = model_rgb.to('cuda')
        images = images.to('cuda')
        summary(model_rgb,(3, 352, 352))

    