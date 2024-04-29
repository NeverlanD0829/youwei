import torch.nn as nn
import torchvision.models as models
import torch.hub

BATCH_SIZE = 10
num_classes = 1

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        # 加载预训练的MobileNetV2模型，但不加载权重
        self.mobilenet = torch.hub.load('pytorch/vision', 'mobilenet_v2', weights=None)   #models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # 修改输入通道数，MobileNetV2默认接受3通道的图像，如果您的输入是单通道的灰度图像，需要进行相应的修改
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # 修改最后的全连接层以适应你的分类任务
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)


