from collections import OrderedDict

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.datasets import datasets
from src.models import models
from src.utils import utils, scheduler, memory


class PretrainClipSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset, self.val_dataset = datasets.get_datasets(
            config.dataset.root, config.dataset.dataset)

        self.train_labels = self.train_dataset.get_targets()

        self.image_encoder = models.get_image_encoder(
            config.model.image_encoder,
            self.config.model.low_dim,
        )

        self.hash_encoder = models.get_hash_encoder(
            config.model.hash_encoder,
            self.config.model.low_dim,
        )

        self.temperature = config.loss.temperature

        # only used for evaluation
        self.image_memory_bank = memory.MemoryBank(
            len(self.train_dataset), 
            self.config.model.low_dim,
        )
        self.hash_memory_bank = memory.MemoryBank(
            len(self.train_dataset), 
            self.config.model.low_dim,
        )

    def configure_optimizers(self):
        # https://github.com/Zasder3/train-CLIP
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
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
            warmup_steps=2000
        )
        return [optimizer], [schedule]

    def get_loss(self, batch, train=True):
        indices, images, hash_input_ids, hash_attention_mask, _, = batch
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
        _, images, hash_input_ids, hash_attention_mask, label, = batch
        _, image_projs = self.image_encoder(images)
        _, hash_projs = self.hash_encoder(
            input_ids=hash_input_ids, attention_mask=hash_attention_mask)

        image_all_dps = self.image_memory_bank.get_all_dot_products(image_projs)
        _, image_nei_idxs = torch.topk(image_all_dps, k=1, sorted=False, dim=1)
        image_nei_idxs = image_nei_idxs.squeeze(1).cpu().numpy()
        image_nei_labels = self.train_ordered_labels[image_nei_idxs]
        image_nei_labels = torch.from_numpy(image_nei_labels).long()
        image_num_correct = torch.sum(image_nei_labels.cpu() == label.cpu()).item()

        hash_all_dps = self.hash_memory_bank.get_all_dot_products(hash_projs)
        _, hash_nei_idxs = torch.topk(hash_all_dps, k=1, sorted=False, dim=1)
        hash_nei_idxs = hash_nei_idxs.squeeze(1).cpu().numpy()
        hash_nei_labels = self.train_ordered_labels[hash_nei_idxs]
        hash_nei_labels = torch.from_numpy(hash_nei_labels).long()
        hash_num_correct = torch.sum(hash_nei_labels.cpu() == label.cpu()).item()

        total_size = images.size(0)

        return image_num_correct, hash_num_correct, total_size

    def training_step(self, batch, _):
        loss = self.get_loss(batch, train=True)
        self.log_dict({'loss': loss} )
        return loss

    def validation_step(self, batch, _):
        loss = self.get_loss(batch, train=False)
        image_num_correct, hash_num_correct, batch_size = \
            self.get_nearest_neighbor_label(batch)
        output = OrderedDict({
            'val_loss': loss,
            'val_image_num_correct': image_num_correct,
            'val_hash_num_correct': hash_num_correct,
            'val_num_total': batch_size,
        })
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
