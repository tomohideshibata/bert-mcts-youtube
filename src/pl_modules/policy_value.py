from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchmetrics
from transformers import AdamW, get_linear_schedule_with_warmup

from src.data.policy_value import PolicyValueDataset
from src.model.bert import BertPolicyValue


class PolicyValueModule(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        # self.hparams = hparams
        if hparams is not None:
            self.hparams.update(hparams)
        self.model = BertPolicyValue(self.hparams['model_dir'] if hparams is not None else None)
        self.val_acc_policy = torchmetrics.Accuracy()

    def forward(self, input_ids, labels=None):
        output = self.model(input_ids, labels)
        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch.pop('input_ids')
        output = self(input_ids, batch)
        return {'loss': output['loss']}

    def validation_step(self, batch, batch_idx):
        input_ids = batch.pop('input_ids')
        output = self(input_ids, batch)
        self.val_acc_policy.update(output['policy_logits'], output['labels'])        
        for k, v in output.items():
            if k == "policy_logits" or k == "labels":
                continue
            output[k] = v.detach().cpu().numpy()
        return output

    def validation_epoch_end(self, outputs):
        val_loss = np.mean([out['loss'] for out in outputs])
        val_loss_policy = np.mean([out['loss_policy'] for out in outputs])
        val_loss_value = np.mean([out['loss_value'] for out in outputs])
        self.log('val_loss', val_loss)
        self.log('val_loss_policy', val_loss_policy)
        self.log('val_loss_value', val_loss_value)
        self.log('val_acc_policy', self.val_acc_policy.compute(), prog_bar=True)
        self.val_acc_policy.reset()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams["train_params"]["weight_decay"],
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams["train_params"]["learning_rate"],
                          eps=self.hparams["train_params"]["adam_epsilon"])

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams["train_params"]["warmup_steps"],
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]        


class PolicyValueDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        dataset_dir = Path(self.cfg.dataset_dir)
        train_data = np.load(dataset_dir / 'train.npy', allow_pickle=True)
        print('Load train data')
        valid_data = np.load(dataset_dir / 'val.npy', allow_pickle=True)
        print('Load val data')
        self.train_dataset = PolicyValueDataset(train_data)
        self.val_dataset = PolicyValueDataset(valid_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.train_loader)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.cfg.val_loader)

