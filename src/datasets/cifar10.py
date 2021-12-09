import hashlib
import numpy as np
import torch.utils.data as data
from torchvision import datasets, transforms


class CIFAR10(data.Dataset):

    def __init__(self, root, train=True, image_transforms=None):
        super().__init__()
        self.dataset = datasets.cifar.CIFAR10(
            root, 
            train=train,
            download=True,
            transform=image_transforms,
        )

    def get_targets(self):
        return np.array(self.dataset.targets)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)


class HashedCIFAR10(CIFAR10):

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        # hash the transformed image
        bytes = transforms.toPIL()(image).tobytes()
        hash = hashlib.sha256(bytes).hexdigest()
        hash = list(hash)  # list of chars

        return image, hash, label
