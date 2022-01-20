#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: datasets.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import numpy as np
from torch.utils.data import Dataset


class SingleDataset(Dataset):
    def __init__(self, data_file, feats_dir, sep=".", protein_level=True,
                 fold_label_file=None):
        # Initialize data
        data = np.loadtxt(data_file, dtype=str)
        self.names = list(data[:, 0])
        self.lengths = data[:, 1].astype("int")
        self.feats_dir = feats_dir
        self.protein_level = protein_level

        # Convert folds to labels (only for training)
        self.folds = data[:, 3]
        self.labels = self._get_labels(fold_label_file, self.folds)

    def _get_labels(self, input_file, level_labels):
        if input_file is not None:
            # Load relation and return numerical labels (training)
            labels = np.loadtxt(input_file, dtype=str)
            label_dict = {item[0]: int(item[1]) for item in labels[1:]}
            return np.array([label_dict[f] for f in level_labels])
        else:
            # Return zeros (testing)
            return np.zeros((len(self.names)), dtype=int)

    def __len__(self):
        # Get total number of samples
        return len(self.names)

    def __getitem__(self, index):
        # Load sample (name, length and label)
        name = self.names[index]
        seq_len = self.lengths[index]
        label = self.labels[index]

        # Load embedding feature matrix
        x = np.load("%s/%s.npy" % (self.feats_dir, name))  # length x dim

        # Get protein level
        if self.protein_level:
            x = x.mean(0)   # dim
        elif x.shape[0] != seq_len:
            seq_len = x.shape[0]    # limited sequence length in ESM embeddings

        return x, label, seq_len, name
