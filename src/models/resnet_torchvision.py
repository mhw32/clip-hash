from torchvision import models


def resnet18_pretrained(low_dim=128, in_channel=3):
    assert in_channel == 3, "pretrained models must be channel 3"
    model = models.resnet18(pretrained=True)


def resnet50_pretrained(low_dim=128, in_channel=3):
    assert in_channel == 3, "pretrained models must be channel 3"
    model = models.resnet50(pretrained=True)
