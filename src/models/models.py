from src.models.logreg import LogisticRegression
from src.models.resnet import resnet18, resnet50
from src.models.resnet_small import resnet18_small, resnet50_small
from src.models.bert import DebertaV3ForSSL


IMAGE_ENCODER = {
    'resnet-18': resnet18,
    'resnet-50': resnet50,
    'resnet-18-small': resnet18_small,
    'resnet-50-small': resnet50_small,
    'logreg': LogisticRegression,
}


HASH_ENCODER = {
    'deberta-v3-small': 'small',
    'deberta-v3-xsmall': 'xsmall',
    'deberta-v3-base': 'base',
    'deberta-v3-large': 'large',
}


def get_image_encoder(model_name, low_dim=128):
    encoder = IMAGE_ENCODER[model_name](low_dim=low_dim, in_channel=3)
    return encoder


def get_hash_encoder(model_name, low_dim=128):
    encoder = DebertaV3ForSSL(
        ow_dim=low_dim, model=HASH_ENCODER[model_name], hidden_dropout_prob=0)
    return encoder


def get_linear_evaluator(low_dim, num_class):
    return LogisticRegression(low_dim, num_class)
