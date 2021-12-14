import hashlib
import numpy as np
import torch.utils.data as data
from torchvision import datasets, transforms
from src.utils.tokenizer import DebertaV3Tokenizer
from transformers import RobertaTokenizer


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
        bert_model='deberta-v3-base',
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
        self.bert_model = bert_model

    def __getitem__(self, index):
        output = super().__getitem__(index)
        """
        # hash the transformed image
        bytes = transforms.ToPILImage()(output['images']).tobytes()
        hash = hashlib.sha256(bytes).hexdigest()
        tokenized = self.tokenizer.tokenize(hash, self.max_seq_len)
        output.update(tokenized)
        """
        if 'deberta' in self.bert_model:
            tokenized = self.tokenizer.tokenize(str(output['labels']), self.max_seq_len)
        elif 'roberta' in self.bert_model:
            tokenized = self.tokenizer(
                str(output['labels']), truncation=True, padding='max_length', 
                max_length=self.max_seq_len, pad_to_max_length=True, return_tensors='pt')
            for k in tokenized.keys():
                tokenized[k] = tokenized[k].squeeze(0)
        output.update(tokenized)

        return output


class MultimodalCIFAR10(HashedCIFAR10):

    pass
