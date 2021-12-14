import os
from src.models.logreg import LogisticRegression
from src.models.resnet import resnet18, resnet50
from src.models.resnet_small import resnet18_small, resnet50_small
from src.models.resnet_mod import resnet18_mod, resnet50_mod
from src.models.resnet_torchvision import resnet18_pretrained, resnet50_pretrained
from src.models.bert import DebertaV3ForSSL, RobertaForSSL


IMAGE_ENCODER = {
    'resnet-18': resnet18,
    'resnet-50': resnet50,
    'resnet-18-small': resnet18_small,
    'resnet-50-small': resnet50_small,
    'resnet-18-mod': resnet18_mod,
    'resnet-50-mod': resnet50_mod,
    'resnet-18-pretrained': resnet18_pretrained,
    'resnet-50-pretrained': resnet50_pretrained,
    'logreg': LogisticRegression,
}

DEBERTA_ROOT = '/data2/wumike/clip_hash'

HASH_ENCODER = {
    'deberta-v3-small': 'deberta-v3-small',
    'deberta-v3-xsmall': 'deberta-v3-xsmall',
    'deberta-v3-base': 'deberta-v3-base',
    'deberta-v3-large': 'deberta-v3-large',
    'roberta-base': 'roberta-base',
}


def get_image_encoder(model_name, low_dim=128, trainable=True):
    encoder = IMAGE_ENCODER[model_name](low_dim=low_dim, in_channel=3)
    for p in encoder.parameters():
        p.requires_grad = trainable
    return encoder


def get_hash_encoder(model_name, low_dim=128, finetune_layers=3, trainable=True):
    if 'deberta' in model_name:
        encoder = DebertaV3ForSSL(low_dim=low_dim, model=HASH_ENCODER[model_name])
        for p in encoder.parameters():
            p.requires_grad = trainable
    elif 'roberta' in model_name:
        encoder = RobertaForSSL(low_dim=low_dim, model=HASH_ENCODER[model_name])
        for p in encoder.parameters():
            p.requires_grad = False
        for p in encoder.roberta.pooler.parameters():
            p.requires_grad = trainable
        for param in encoder.roberta.encoder.layer[-finetune_layers:].parameters():
            param.requires_grad = trainable
    else:
        raise Exception(f'Model name {model_name} not supported.')

    return encoder


def get_linear_evaluator(low_dim, num_class):
    return LogisticRegression(low_dim, num_class)
