#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: trainers.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
from pytorch_lightning import LightningModule
from utils_model.lmcl import LMCosineLoss


class BaseTrain(LightningModule, ABC):
    def __init__(self, net, hparams, log_file):
        super().__init__()
        self.net = net
        self.hparams = hparams
        self.log_file = log_file

    def configure_optimizers(self):
        ''' Define optimizer and scheduler for learning rate adjustment '''
        optimizer = Adam(self.net.parameters(), lr=self.hparams.init_lr,
                         weight_decay=self.hparams.weight_decay)
        if self.hparams.lr_sched:
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, milestones=self.hparams.lr_steps
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def on_epoch_start(self):
        ''' Print epoch number at the start of each epoch '''
        with open(self.log_file, "a") as f:
            print("[*] Epoch %d..." % (self.current_epoch+1), file=f)

    def training_epoch_end(self, train_outs):
        ''' Print/Log training loss and metrics at the end of each epoch '''
        self._shared_epoch_end(train_outs, "train")

    def validation_epoch_end(self, valid_outs):
        ''' Print/Log validation loss and metrics at the end of each epoch '''
        self._shared_epoch_end(valid_outs, "valid")

    @abstractmethod
    def _shared_epoch_end(self, outputs, set_name):
        pass

    def _print_file_epoch(self, set_name, tag, metric, metric_std=None):
        ''' Print metric (avg/std) to file at the end of each epoch '''
        with open(self.log_file, "a") as f:
            if metric_std is None:
                print("--- Average %s %s: %.4f" % \
                      (set_name, tag, metric), file=f)
            else:
                print("--- Average %s %s: %.4f +/- %.4f" % \
                      (set_name, tag, metric, metric_std), file=f)

    def _log_tb_epoch(self, set_name, tag, metric):
        ''' Log metric to Tensorboard at the end of each epoch '''
        self.logger.experiment.add_scalar(
            "epoch_%s/%s_%s" % (set_name, set_name, tag),
            metric, global_step=self.current_epoch+1
        )


class MultiClassTrain(BaseTrain):
    def __init__(self, net, hparams, log_file):
        super().__init__(net, hparams, log_file)
        self.net = net
        self.hparams = hparams
        self.log_file = log_file
        self.criterion = (LMCosineLoss(margin=self.hparams.loss_margin,
                                       scale=self.hparams.loss_scale)
                          if ("lmcl" in self.hparams.loss_type)
                          else CrossEntropyLoss())

    def _shared_epoch_end(self, outputs, set_name):
        ''' Print/Log the average loss and accuracy (top 1 and top 5) values
            at each epoch '''
        for tag in ["loss", "acc", "acc_top5"]:
            metric = torch.Tensor([item[tag] for item in outputs]).mean()
            # Print in file
            self._print_file_epoch(set_name, tag="class "+tag, metric=metric)
            # Log to Tensorboard
            if tag != "acc_top5":
                self._log_tb_epoch(set_name, tag="class_"+tag, metric=metric)

    def shared_step(self, batch):
        ''' Perform a shared forward step, calculate multiclass loss and
            classification accuracy (top 1 and top 5) '''
        labels = batch.y
        # Forward pass
        _, outputs = self.net(batch)
        # Calculate loss
        loss = self.criterion(outputs, labels)
        # Get classification accuracy
        preds = torch.argmax(outputs, dim=1)
        accuracy = torch.mean((preds == labels).float())
        # Get top 5 accuracy
        _, preds_top5 = torch.topk(outputs, k=5)
        accuracy_top5 = torch.mean(
            (preds_top5 == labels.unsqueeze(1)).any(1).float()
        )
        return loss, accuracy, accuracy_top5

    def training_step(self, batch, _):
        ''' Perform a training step '''
        loss, accuracy, accuracy_top5 = self.shared_step(batch)
        # Log iteration
        self.log_dict({"step_train_class_loss": loss,
                       "step_train_class_acc": accuracy})
        return {"loss": loss, "acc": accuracy, "acc_top5": accuracy_top5}

    def validation_step(self, batch, _):
        ''' Perform a validation step '''
        loss, accuracy, accuracy_top5 = self.shared_step(batch)
        return {"loss": loss, "acc": accuracy, "acc_top5": accuracy_top5}

    def test_step(self, sample, _):
        ''' Perform a test step, extract embeddings/predictions '''
        # Get sample identifier as string
        name = "".join(sample.name)
        # Forward pass
        embeddings, outputs = self.net(sample)
        _, predictions = torch.topk(outputs, k=5)    # get top 5 predictions
        # Return output embedding vector
        return {"name": name, "emb": embeddings.cpu().numpy().flatten(),
                "pred": predictions.cpu().numpy().flatten(),
                "pred_logits": outputs.cpu().numpy().flatten()}

    def test_epoch_end(self, test_outs):
        ''' Generate the test embeddings/predictions dictionaries '''
        self.embeddings = {item["name"]: item["emb"] for item in test_outs}
        self.predictions = {item["name"]: item["pred"] for item in test_outs}
        self.pred_logits = {item["name"]: item["pred_logits"] for item in test_outs}
