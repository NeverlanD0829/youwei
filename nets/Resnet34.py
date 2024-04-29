import torch.nn as nn
import torchvision.models as models

BATCH_SIZE = 10
num_classes = 1
class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()

        # 加载预训练的ResNet-34模型，但不加载权重
        self.resnet34 = models.resnet34(weights=None)

        # 修改第一个卷积层以适应你的输入数据（例如，单通道灰度图像）
        self.resnet34.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改最后的全连接层以适应你的分类任务
        num_features = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.resnet34(x)


# 创建自定义ResNet-34模型，设置类别数量

