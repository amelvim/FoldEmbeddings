#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: collate_functions.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import numpy as np
import torch
from utils_data.data import Data


def single_collate(batch):
    # Get features, label, length and name (from a list of arrays)
    (x, y, lens, names) = zip(*batch)

    # Convert to tensors and return Data object
    return Data(x=torch.from_numpy(np.array(x)),
                y=torch.from_numpy(np.array(y)),
                seq_len=torch.from_numpy(np.array(lens)),
                name=np.array(names))


def single_sequence_collate(batch):
    # Get features, label, length and name (from a list of arrays)
    (x, y, lens, names) = zip(*batch)

    # Pad features to max sequence length in the batch
    max_len = max(lens)
    x_pad = [np.pad(item, ((0, max_len-lens[i]), (0, 0)), "constant") \
             for i, item in enumerate(x)]

    # Create mask for each sample
    masks = [np.expand_dims([1]*i + [0]*(max_len - i), axis=0) for i in lens]

    # Convert to tensors and return Data object
    return Data(x=torch.from_numpy(np.array(x_pad)),
                y=torch.from_numpy(np.array(y)),
                seq_len=torch.from_numpy(np.array(lens)),
                seq_mask=torch.from_numpy(np.array(masks, dtype=np.float32)),
                name=np.array(names))
