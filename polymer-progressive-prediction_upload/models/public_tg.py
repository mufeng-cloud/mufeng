"""Chemprop encoder with Log10 top-three digit MSE only; no reconstruction loss."""
import torch
from torch import nn
from lightning import pytorch as pl
from chemprop import nn as cpnn
from torchmetrics import MeanSquaredError

from models.public_tg_head import ProgressiveTgHead, weighted_digit_mse, encode_digits
from configs.public_tg_protocol import MAX_EPOCHS


class ProgressiveTgModel(pl.LightningModule):
    def __init__(self,dropout=0.0,weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.message_passing = cpnn.BondMessagePassing()
        self.agg = cpnn.MeanAggregation()
        self.bn = nn.BatchNorm1d(300)
        self.head = ProgressiveTgHead(dropout=dropout)
        self.weight_decay = float(weight_decay)
        # AdamW with one non-restarting cosine decay over the fixed epoch budget.
        self.warmup_epochs = 0
        self.init_lr, self.max_lr, self.final_lr = 2e-3, 2e-3, 1e-7
        self.cosine_t_max = MAX_EPOCHS
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)

    def forward(self, bmg, V_d=None, X_d=None, targets=None):
        if X_d is not None:
            raise ValueError('This experiment uses no extra descriptors')
        hidden = self.bn(self.agg(self.message_passing(bmg, V_d), bmg.batch))
        return self.head(hidden, targets)

    def training_step(self, batch, batch_idx):
        bmg, V_d, X_d, targets, *_ = batch
        predictions, _ = self(bmg, V_d, X_d, targets)
        digit_targets=encode_digits(targets)[1]
        loss=weighted_digit_mse(predictions,digit_targets)
        if not hasattr(self,'loss_audit'):
            self.loss_audit={'targets':targets.detach().reshape(-1).cpu().tolist(),'raw_digits':predictions.detach().cpu().tolist(),'digit_targets':digit_targets.detach().cpu().tolist(),'digit_loss':float(loss.detach()),'reconstruction_loss_coefficient':0.,'auxiliary_reconstruction_removed':True,'loss':float(loss.detach())}
        self.log('train_digit_mse',loss,batch_size=targets.shape[0],on_step=False,on_epoch=True)
        self.log('train_total_loss',loss,batch_size=targets.shape[0],on_step=False,on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        bmg, V_d, X_d, targets, *_ = batch
        _, predictions = self(bmg, V_d, X_d)
        self.val_rmse.update(predictions, targets)
        self.log('val_rmse_log10', self.val_rmse, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        bmg, V_d, X_d, targets, *_ = batch
        _, predictions = self(bmg, V_d, X_d)
        self.test_rmse.update(predictions, targets)
        self.log('test_rmse_log10', self.test_rmse, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        bmg, V_d, X_d, *_ = batch
        digits, log10 = self(bmg, V_d, X_d)
        return {'digits': digits, 'log10': log10}

    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.init_lr,weight_decay=self.weight_decay,
                                    betas=(0.9,0.999),eps=1e-8)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.cosine_t_max,
                                                            eta_min=self.final_lr)
        return {'optimizer':optimizer,'lr_scheduler':{'scheduler':scheduler,'interval':'epoch','frequency':1}}
