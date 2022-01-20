#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: callbacks.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import pickle
from pytorch_lightning.callbacks import Callback, ModelCheckpoint


class MyModelCheckpoint(ModelCheckpoint):
    def on_validation_end(self, trainer, pl_module):
        pass
    def on_epoch_end(self, trainer, pl_module):
        self.save_checkpoint(trainer, pl_module)


class TestSerializer(Callback):
    def __init__(self, emb_path=None, pred_path=None, pred_logits_path=None):
        self.emb_path = emb_path
        self.pred_path = pred_path
        self.pred_logits_path = pred_logits_path
    def on_test_end(self, trainer, pl_module):
        if self.emb_path is not None:
            with open(self.emb_path, "wb") as f:
                pickle.dump(pl_module.embeddings, f)
        if self.pred_path is not None:
            with open(self.pred_path, "wb") as f:
                pickle.dump(pl_module.predictions, f)
        if self.pred_logits_path is not None:
            with open(self.pred_logits_path, "wb") as f:
                pickle.dump(pl_module.pred_logits, f)
