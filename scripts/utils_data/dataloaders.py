#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: dataloaders.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import torch
from torch.utils.data import DataLoader
from utils_data.datasets import SingleDataset
from utils_data.collate_functions import single_collate, single_sequence_collate


def define_collate(average=False):
    ''' Method to define the collate function used in the dataloader '''
    return single_collate if average else single_sequence_collate


def get_train_loader(args):
    ''' Method to get the train dataset and dataloader '''
    # Define dataset
    train_set = SingleDataset(
        args.train_file, args.feats_dir, sep=args.scop_separation,
        protein_level=args.protein_level, fold_label_file=args.fold_label_file
    )
    # Define collate function
    collate_fn = define_collate(average=args.protein_level)
    # Define dataloader
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size_class, shuffle=True,
        drop_last=True, num_workers=args.ndata_workers, collate_fn=collate_fn
    )
    return train_loader


def get_valid_loader(args):
    ''' Method to get the validation dataset and dataloader '''
    # Define dataset
    valid_set = SingleDataset(
        args.valid_file, args.feats_dir, sep=args.scop_separation,
        protein_level=args.protein_level, fold_label_file=args.fold_label_file
    )
    # Define collate function
    collate_fn = define_collate(average=args.protein_level)
    # Define dataloader
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size_class,
        num_workers=args.ndata_workers, collate_fn=collate_fn
    )
    return valid_loader


def get_test_loader(args):
    ''' Method to get the test dataset and dataloader '''
    # Define dataset
    test_set = SingleDataset(
        args.test_file, args.feats_dir_test, sep=args.scop_separation,
        protein_level=args.protein_level, fold_label_file=None
    )
    # Define collate function
    collate_fn = define_collate(average=args.protein_level)
    # Define dataloader
    test_loader = DataLoader(
        test_set, batch_size=1, num_workers=args.ndata_workers,
        collate_fn=collate_fn
    )
    return test_loader
