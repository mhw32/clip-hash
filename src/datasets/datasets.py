import random
from PIL import ImageFilter
from torchvision import transforms
from src.datasets.cifar10 import CIFAR10, HashedCIFAR10, MultimodalCIFAR10
from src.datasets.imagenet import ImageNet, HashedImageNet, MultimodalImageNet

DATASET = {
    'cifar10': CIFAR10,
    'hashed_cifar10': HashedCIFAR10,
    'multimodal_cifar10': MultimodalCIFAR10,
    'imagenet': ImageNet,
    'hashed_imagenet': HashedImageNet,
    'multimodal_imagenet': MultimodalImageNet,
}

NUM_CLASS_DICT = {
    'cifar10': 10, 
    'hashed_cifar10': 10,
    'multimodal_cifar10': 10,
    'imagenet': 1000,
    'hashed_imagenet': 1000,
    'multimodal_imagenet': 1000,
}


def load_imagenet_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            GaussianBlur(sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, test_transforms


def load_cifar10_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            GaussianBlur(sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, test_transforms


TRANSFORM = {
    'cifar10': load_cifar10_transforms,
    'hashed_cifar10': load_cifar10_transforms,
    'multimodal_cifar10': load_cifar10_transforms,
    'imagenet': load_imagenet_transforms,
    'hashed_imagenet': load_imagenet_transforms,
    'multimodal_imagenet': load_imagenet_transforms,
}


def get_datasets(root, dataset_name):
    """
    Master function for loading datasets and toggle between
    different image transformation.
    """
    train_transforms, test_transforms = TRANSFORM[dataset_name]()
    train_dataset = DATASET[dataset_name](
        root, train=True, image_transforms=train_transforms)
    val_dataset = DATASET[dataset_name](
        root, train=False, image_transforms=test_transforms)
    train_dataset.num_class = NUM_CLASS_DICT[dataset_name]
    val_dataset.num_class = NUM_CLASS_DICT[dataset_name]
    return train_dataset, val_dataset


def undo_imagenet_transforms():
    return transforms.Compose([
        UnNormalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
        transforms.ToPIL(),
    ])


class GaussianBlur(object):

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class UnNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
