#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: lmcl.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import torch
import torch.nn as nn
import torch.nn.functional as F


class LMCosineLoss(nn.Module):
    ''' CosFace Large Margin Cosine Loss:
        Softmax loss as a cosine loss by L2 normalization of features
        and weight vectors. Adapted from:
        https://github.com/YirongMao/softmax_variants/blob/master/model_utils.py
    '''
    def __init__(self, margin=0.2, scale=1):
        super().__init__()
        # Define margin and scale hyperparameters
        self.margin = margin
        self.scale = scale

    def forward(self, logits, labels):
        # Create margin one-hot vectors
        vec_margin = torch.zeros(logits.shape, device=logits.device)
        vec_margin.scatter_(1, labels.unsqueeze(dim=-1), self.margin)
        # Substract margin only for the correct class and multiply by scale
        margin_logits = self.scale * (logits - vec_margin)
        # Return cross-entropy loss between logits with margin/scale and labels
        return F.cross_entropy(margin_logits, labels)


class ModelLMCL(nn.Module):
    def __init__(self, net, hparams):
        super().__init__()
        self.net = net
        # Set old classification layer to Identity()
        self.net.out_layer = nn.Identity()
        # Create a new output matrix (centroids to be learned)
        self.centroids = nn.Parameter(
            torch.randn(hparams.num_classes, hparams.hidden_dims[-1])
        )

    def _apply_l2_norm(self, x):
        # Calculate L2 norm of a vector and apply normalization
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return torch.div(x, norm)

    def forward(self, inp):
        # Get embedding vectors from the network model
        emb, _ = self.net(inp)
        # Apply L2 normalization to the embedding vectors and centroids
        emb_norm = self._apply_l2_norm(emb)
        cent_norm = self._apply_l2_norm(self.centroids)
        # Get output logits
        out = torch.matmul(emb_norm, torch.transpose(cent_norm, 0, 1))
        return emb, out
