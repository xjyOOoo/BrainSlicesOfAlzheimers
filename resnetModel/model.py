import torch.nn as nn
import torchvision

# 定义ResNet模型
class ResNetModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetModel, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
