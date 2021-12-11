import hashlib
import numpy as np
import torch.utils.data as data
from torchvision import datasets, transforms
from src.utils.tokenizer import DebertaV3Tokenizer


class CIFAR10(data.Dataset):

    def __init__(self, root, train=True, image_transforms=None, **kwargs):
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
        image, label = self.dataset.__getitem__(index)
        output = dict(indices=index, images=image, labels=label)
        return output

    def __len__(self):
        return len(self.dataset)


class HashedCIFAR10(CIFAR10):

    def __init__(
        self,
        root,
        deberta_model='base',
        train=True,
        image_transforms=None,
        max_seq_len=512,
    ):
        super().__init__(root, train=train, image_transforms=image_transforms)
        self.tokenizer = DebertaV3Tokenizer(deberta_model)
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        output = super().__getitem__(index)
        # hash the transformed image
        bytes = transforms.ToPILImage()(output['images']).tobytes()
        hash = hashlib.sha256(bytes).hexdigest()
        hash = ' '.join(list(hash))
        tokenized = self.tokenizer.tokenize(hash, self.max_seq_len)
        output.update(tokenized)

        return output


class MultimodalCIFAR10(HashedCIFAR10):

    def __getitem__(self, index):
        output = super().__getitem__(index)

        bytes = transforms.ToPILImage()(output['images']).tobytes()
        hash = hashlib.sha256(bytes).hexdigest()
        hash = ' '.join(list(hash))
        tokenized = self.tokenizer.tokenize(hash, self.max_seq_len)
        output.update(tokenized)

        return output
