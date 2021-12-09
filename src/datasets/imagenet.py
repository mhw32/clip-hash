import os, hashlib
import numpy as np
import torch.utils.data as data
from torchvision import datasets, transforms


class ImageNet(data.Dataset):

    def __init__(self, root, train=True, image_transforms=None):
        super().__init__()
        split_dir = 'train' if train else 'validation'
        imagenet_dir = os.path.join(root, split_dir)
        self.dataset = datasets.ImageFolder(imagenet_dir, image_transforms)

    def get_targets(self):
        return np.array(self.dataset.targets)

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        return index, image, label

    def __len__(self):
        return len(self.dataset)


class HashedImageNet(ImageNet):

    def __getitem__(self, index):
        _, image, label = super().__getitem__(index)
        # hash the transformed image
        bytes = transforms.toPIL()(image).tobytes()
        hash = hashlib.sha256(bytes).hexdigest()
        return index, image, hash, label

