import torch.nn as nn
from torchvision import models
from src.models.projection import ProjectionHead


class PretrainedResNet(nn.Module):

    def __init__(self, depth, low_dim=128, in_channel=3):
        super().__init__()

        if depth == 18:
            resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)

        del resnet.fc; resnet.fc = lambda x: x
        self.resnet = resnet
        self.projection_head = ProjectionHead(512, low_dim)

    def forward(self, x):
        embedding = self.resnet(x)
        projection = self.projection_head(embedding)
        return embedding, projection


def resnet18_pretrained(low_dim=128, in_channel=3):
    assert in_channel == 3, "pretrained models must be channel 3"
    return PretrainedResNet(18, low_dim=low_dim, in_channel=in_channel)


def resnet50_pretrained(low_dim=128, in_channel=3):
    assert in_channel == 3, "pretrained models must be channel 3"
    return PretrainedResNet(50, low_dim=low_dim, in_channel=in_channel)

