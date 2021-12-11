import os
from itertools import chain
from dotmap import DotMap

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl

from src.datasets import datasets
from src.models import models
from src.utils import utils, scheduler, memory


class PretrainClipSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset, self.val_dataset = datasets.get_datasets(
            config.dataset.root, 
            config.dataset.dataset, 
            config.model.hash_encoder,
            max_seq_len=config.dataset.max_seq_len,
        )

        self.train_labels = self.train_dataset.get_targets()

        self.image_encoder = models.get_image_encoder(
            config.model.image_encoder,
            config.model.low_dim,
            trainable=True,
        )

        self.hash_encoder = models.get_hash_encoder(
            config.model.hash_encoder,
            config.model.low_dim,
            trainable=True,
        )

        self.temperature = config.loss.temperature

        # only used for evaluation
        self.image_memory_bank = memory.MemoryBank(
            len(self.train_dataset), 
            config.model.low_dim,
        )
        self.hash_memory_bank = memory.MemoryBank(
            len(self.train_dataset), 
            config.model.low_dim,
        )

    def configure_optimizers(self):
        # https://github.com/Zasder3/train-CLIP
        optimizer = torch.optim.AdamW(
            chain(
                self.image_encoder.parameters(),
                self.hash_encoder.parameters(),
            ),
            lr=self.config.optimizer.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.2,
        )
        schedule = scheduler.CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.num_training_steps,
            cycle_mult=1.0,
            max_lr=self.config.optimizer.lr,
            min_lr=0,
            warmup_steps=self.config.optimizer.warmup_steps,
        )
        return [optimizer], [schedule]

    def get_loss(self, batch, train=True):
        indices = batch['indices']
        images = batch['images']
        hash_input_ids = batch['input_ids']
        hash_attention_mask = batch['attention_mask']

        _, image_projs = self.image_encoder(images)
        _, hash_projs = self.hash_encoder(
            input_ids=hash_input_ids, attention_mask=hash_attention_mask)

        image_projs = utils.l2_normalize(image_projs)
        hash_projs = utils.l2_normalize(hash_projs)

        logits = (image_projs @ hash_projs.t()) / self.temperature
        labels = torch.arange(logits.size(0)).type_as(logits).long()
        image_loss = F.cross_entropy(logits, labels, reduction='none')
        hash_loss = F.cross_entropy(logits.t(), labels, reduction='none')
        loss = (image_loss + hash_loss) / 2.0

        if train:  # save embeddings into memory bank
            with torch.no_grad():
                self.image_memory_bank.update(indices, image_projs)
                self.hash_memory_bank.update(indices, hash_projs)

        return loss.mean()

    @torch.no_grad()
    def get_nearest_neighbor_label(self, batch):
        """
        NOTE: ONLY TO BE USED FOR VALIDATION.

        For each image in validation, find the nearest image in the 
        training dataset using the memory bank. Assume its label as
        the predicted label.
        """
        images = batch['images']
        hash_input_ids = batch['input_ids']
        hash_attention_mask = batch['attention_mask']
        labels = batch['labels']

        _, image_projs = self.image_encoder(images)
        _, hash_projs = self.hash_encoder(
            input_ids=hash_input_ids, attention_mask=hash_attention_mask)

        image_all_dps = self.image_memory_bank.get_all_dot_products(image_projs)
        _, image_nei_idxs = torch.topk(image_all_dps, k=1, sorted=False, dim=1)
        image_nei_idxs = image_nei_idxs.squeeze(1).cpu().numpy()
        image_nei_labels = self.train_ordered_labels[image_nei_idxs]
        image_nei_labels = torch.from_numpy(image_nei_labels).long()
        image_num_correct = torch.sum(image_nei_labels.cpu() == labels.cpu()).item()

        hash_all_dps = self.hash_memory_bank.get_all_dot_products(hash_projs)
        _, hash_nei_idxs = torch.topk(hash_all_dps, k=1, sorted=False, dim=1)
        hash_nei_idxs = hash_nei_idxs.squeeze(1).cpu().numpy()
        hash_nei_labels = self.train_ordered_labels[hash_nei_idxs]
        hash_nei_labels = torch.from_numpy(hash_nei_labels).long()
        hash_num_correct = torch.sum(hash_nei_labels.cpu() == labels.cpu()).item()

        total_size = images.size(0)

        return image_num_correct, hash_num_correct, total_size

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps:
            return self.trainer.max_steps

        dataset_size = len(dataset)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = \
            dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs

    def training_step(self, batch, _):
        loss = self.get_loss(batch, train=True)
        self.log_dict({'train_loss': loss} )
        return loss

    def validation_step(self, batch, _):
        loss = self.get_loss(batch, train=False)
        image_num_correct, hash_num_correct, batch_size = \
            self.get_nearest_neighbor_label(batch)
        output = {
            'val_loss': loss,
            'val_image_num_correct': image_num_correct,
            'val_hash_num_correct': hash_num_correct,
            'val_num_total': batch_size,
        }
        return output
    
    def validation_epoch_end(self, outputs):
        metrics = {}
        metrics['val_loss'] = torch.tensor(
            [elem['val_loss'] for elem in outputs]).float().mean()
        image_num_correct = sum([out['val_image_num_correct'] for out in outputs])
        hash_num_correct = sum([out['val_hash_num_correct'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_image_acc = image_num_correct / float(num_total)
        val_hash_acc = hash_num_correct / float(num_total)
        metrics['val_image_acc'] = val_image_acc
        metrics['val_hash_acc'] = val_hash_acc
        self.log_dict(metrics)

    def train_dataloader(self):
        return utils.create_dataloader(self.train_dataset, self.config)

    def val_dataloader(self):
        return utils.create_dataloader(self.val_dataset, self.config, shuffle=False)


class BaseTransferSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.image_encoder, self.hash_encoder, self.pretrain_config = \
            self.get_encoders()

        self.train_dataset, self.val_dataset = datasets.get_datasets(
            config.dataset.root, config.dataset.dataset, config.model.hash_encoder)

        self.model = models.get_linear_evaluator(
            self.pretrain_config.model.low_dim, 
            self.train_dataset.num_class,
        )

    def get_encoders(self):
        base_dir = self.config.pretrain.exp_dir
        checkpoint_name = self.config.pretrain.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
        config_json = utils.load_json(config_path)
        config = DotMap(config_json)

        system = PretrainClipSystem(config)
        checkpoint_file = os.path.join(base_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        system.load_state_dict(checkpoint['state_dict'], strict=False)

        image_encoder = system.image_encoder.eval()
        hash_encoder = system.hash_encoder.eval()

        # freeze parameters!
        utils.frozen_params(image_encoder)
        utils.frozen_params(hash_encoder)

        return image_encoder, hash_encoder, config

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            momentum=self.config.optimizer.momentum,
            weight_decay=self.config.optimizer.weight_decay,
        )
        schedule = MultiStepLR(
            optimizer,
            self.config.optimizer.milestones,
            gamma=0.1,
        )
        return [optimizer], [schedule]

    def get_loss(self, batch, train=True):
        raise NotImplementedError

    def training_step(self, batch, _):
        loss, num_correct, num_total = self.get_loss(batch, train=True)
        self.log_dict({
            'train_loss': loss,
            'train_num_correct': num_correct,
            'train_num_total': num_total,
        })
        return loss

    def validation_step(self, batch, _):
        loss, num_correct, num_total = self.get_loss(batch, train=False)
        output = {
            'val_loss': loss,
            'val_num_correct': num_correct,
            'val_num_total': num_total,
        }
        return output
    
    def validation_epoch_end(self, outputs):
        metrics = {}
        metrics['val_loss'] = torch.tensor(
            [elem['val_loss'] for elem in outputs]).float().mean()
        num_correct = sum([out['val_num_correct'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc
        self.log_dict(metrics)

    def train_dataloader(self):
        return utils.create_dataloader(self.train_dataset, self.config)

    def val_dataloader(self):
        return utils.create_dataloader(self.val_dataset, self.config, shuffle=False)


class ImageTransferSystem(BaseTransferSystem):

    def __init__(self, config):
        super().__init__(config)
        del self.hash_encoder

    def get_loss(self, batch, train=True):
        images = batch['images']
        labels = batch['labels']

        _, image_projs = self.image_encoder(images)
        logits = self.model(image_projs)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            preds = preds.long().cpu()
            num_correct = torch.sum(preds == labels.long().cpu())
            num_total = images.size(0)

        return loss, num_correct, num_total


class HashTransferSystem(BaseTransferSystem):

    def __init__(self, config):
        super().__init__(config)
        del self.image_encoder

    def get_loss(self, batch, train=True):
        hash_input_ids = batch['input_ids']
        hash_attention_mask = batch['attention_mask']
        labels = batch['labels']

        _, hash_projs = self.hash_encoder(
            input_ids=hash_input_ids, attention_mask=hash_attention_mask)
        logits = self.model(hash_projs)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            preds = preds.long().cpu()
            num_correct = torch.sum(preds == labels.long().cpu())
            num_total = hash_input_ids.size(0)

        return loss, num_correct, num_total
