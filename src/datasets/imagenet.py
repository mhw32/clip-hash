import os, hashlib
import numpy as np
import torch.utils.data as data
from torchvision import datasets, transforms
from src.utils.tokenizer import DebertaV3Tokenizer


class ImageNet(data.Dataset):

    def __init__(self, root, train=True, image_transforms=None, **kwargs):
        super().__init__()
        split_dir = 'train' if train else 'validation'
        imagenet_dir = os.path.join(root, split_dir)
        self.dataset = datasets.ImageFolder(imagenet_dir, image_transforms)

    def get_targets(self):
        return np.array(self.dataset.targets)

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        output = dict(indices=index, images=image, labels=label)
        return output

    def __len__(self):
        return len(self.dataset)


class HashedImageNet(ImageNet):

    def __init__(
        self,
        root,
        bert_model='base',
        train=True,
        image_transforms=None,
        max_seq_len=512,
    ):
        super().__init__(root, train=train, image_transforms=image_transforms)
        if 'deberta' in bert_model:
            self.tokenizer = DebertaV3Tokenizer(bert_model)
        elif 'roberta' in bert_model:
            self.tokenizer = RobertaTokenizer.from_pretrained(bert_model)
        else:
            raise Exception(f'Model name {bert_model} not supported.')
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        output = super().__getitem__(index)
        # hash the transformed image
        bytes = transforms.ToPILImage()(output['images']).tobytes()
        hash = hashlib.sha256(bytes).hexdigest()
        tokenized = self.tokenizer.tokenize(hash, self.max_seq_len)
        output.update(tokenized)

        del output['images']

        return output


class MultimodalImageNet(HashedImageNet):

    pass
