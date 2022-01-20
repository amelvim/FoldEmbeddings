#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: main_lightning.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import argparse
import pickle
import os

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from lightning.callbacks import MyModelCheckpoint, TestSerializer
from lightning.trainers import MultiClassTrain

from utils_data.dataloaders import (
    get_train_loader, get_valid_loader, get_test_loader
)
from utils_model.lmcl import ModelLMCL
from utils_model.networks import (
    MultiClassMLP, MultiClassResCNNGRU, MultiClassLightAttention
)


def str2bool(value):
    ''' Method to use booleans in argparse '''
    return value.lower() == "true"

def str2list(string):
    ''' Method to convert underscore separated values into list in argparse '''
    return [int(item) for item in string.split("_")]

def create_parser_args():
    ''' Method to create the parser and parse arguments '''
    parser = argparse.ArgumentParser(description="")
    # Common arguments
    parser.add_argument("--ndata_workers", type=int, default=1)
    parser.add_argument(
        "--phase", type=str, default="train", const="train", nargs="?",
        choices=["train", "test"]
    )

    # Training hyperparameters
    parser.add_argument("--batch_size_class", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--loss_margin", type=float, default=0.1) # lmcl
    parser.add_argument("--loss_scale", type=float, default=1) # lmcl
    parser.add_argument("--init_lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lr_sched", type=str2bool, default=True)
    parser.add_argument("--lr_steps", type=str2list, default="40")

    # Architecture hyperparameters
    parser.add_argument("--input_dim", type=int, default=1024)
    parser.add_argument("--num_classes", type=int, default=1154)
    parser.add_argument(
        "--model_type", type=str, default="mlp", const="mlp", nargs="?",
        choices=["mlp", "rescnn_gru", "light_att"]
    )
    parser.add_argument(
        "--loss_type", type=str, default="softmax", const="softmax", nargs="?",
        choices=["softmax", "lmcl"]
    )
    parser.add_argument(
        "--activation", type=str, default="relu", const="relu", nargs="?",
        choices=["relu", "lrelu", "sigmoid", "tanh"]
    )
    parser.add_argument(
        "--activation_last", type=str, default="sigmoid", const="sigmoid",
        nargs="?", choices=["relu", "lrelu", "sigmoid", "tanh"]
    )
    parser.add_argument("--drop_prob", type=float, default=0)
    parser.add_argument("--batch_norm", type=str2bool, default=False)
    # CNN / ResCNN layers
    parser.add_argument("--channel_dims", type=str2list, default="512_1024")
    parser.add_argument("--kernel_sizes", type=str2list, default="5_5")
    # GRU layers
    parser.add_argument("--gru_dim", type=int, default=1024)
    parser.add_argument("--gru_bidirec", type=str2bool, default=True)
    parser.add_argument("--gru_layers", type=int, default=1)
    # MLP layers
    parser.add_argument("--hidden_dims", type=str2list, default="512")

    # Training directories and files
    parser.add_argument("--model_dir", type=str, default="models/seqvec/MLP")
    parser.add_argument("--train_file", type=str,
                        default="data/train/train.list")
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--fold_label_file", type=str,
                        default="data/train/fold_label_relation_1154.txt")
    parser.add_argument("--feats_dir", type=str, default="embeddings/seqvec/train")

    # Test directories and files
    parser.add_argument("--model_file", type=str,
                        default="models/seqvec/MLP/checkpoint/model_epoch80.pth.tar")
    parser.add_argument("--test_file", type=str,
                        default="data/lindahl/lindahl.list")
    parser.add_argument("--feats_dir_test", type=str,
                        default="embeddings/seqvec/lindahl")
    parser.add_argument("--scop_separation", type=str, default="_")
    return parser.parse_args()


def create_dirs(dir_names):
    ''' Method to create non-existing directories '''
    for n in dir_names:
        if not os.path.exists(n):
            os.makedirs(n)

def resume_trainer(ckpt_dir, log_file):
    ''' Method to get the last checkpoint to resume training '''
    try:
        last_ckpt = sorted([ckpt_dir+'/'+f for f in os.listdir(ckpt_dir)],
                            key=os.path.getmtime)[-1]
        with open(log_file, "a") as f:
            print("[*] Loaded checkpoint:", last_ckpt, file=f)
        return (last_ckpt, 0)   # no validation at loaded epoch
    except:
        with open(log_file, "a") as f:
            print("[!] No checkpoint found.", file=f)
        return (None, -1)       # None checkpoint, validation at first epoch

def get_network(model_type):
    ''' Method to get the network model '''
    networks_dict = {
        "mlp": MultiClassMLP,
        "rescnn_gru": MultiClassResCNNGRU,
        "light_att": MultiClassLightAttention
    }
    return networks_dict.get(model_type, "nothing")


def main(args):
    ''' Main method '''
    # Open output file
    log_file = args.model_dir + "/" + args.phase + ".txt"
    with open(log_file, "a") as f:
        print(args, file=f)

    # Initialize network model and set `protein_level` argument
    with open(log_file, "a") as f:
        print("[*] Init %s model..." % args.model_type, file=f)
    args.protein_level = bool("mlp" in args.model_type)
    net = get_network(args.model_type)(args)
    # Extend network model for LMCL loss
    if ("lmcl" in args.loss_type):
        net = ModelLMCL(net, args)

    with open(log_file, "a") as f:
        print("[*] Initialize model successfully.", file=f)
        print(net, file=f)
        print("[*] Number of model parameters:", file=f)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad),
              file=f)

    # Training phase
    if args.phase == "train":

        # Define the Lightning model for the training process
        pl_model = MultiClassTrain(net=net, hparams=args, log_file=log_file)
        # Define Tensorboard logger
        logger = TensorBoardLogger(args.model_dir + "/logs", name="")
        # Create checkpoint directory
        ckpt_dir = args.model_dir + "/checkpoint"
        create_dirs([ckpt_dir])
        # Define checkpoint callback
        ckpt_callback = MyModelCheckpoint(
            filepath=ckpt_dir, save_top_k=-1, period=5
        )
        # Resume trainer if last checkpoint is saved
        last_ckpt, val_start = resume_trainer(ckpt_dir, log_file)
        # Define trainer
        trainer = pl.Trainer(
            gpus=1, logger=logger, max_epochs=args.num_epochs,
            num_sanity_val_steps=val_start, log_save_interval=50,
            checkpoint_callback=ckpt_callback, resume_from_checkpoint=last_ckpt
        )
        # Data loading
        with open(log_file, "a") as f:
            print("[*] Loading training and validation data...", file=f)
        train_loader = get_train_loader(args)
        valid_loader = get_valid_loader(args) if args.valid_file else None
        # Training and validation
        with open(log_file, "a") as f:
            print("[*] Start training...", file=f)
        trainer.fit(pl_model, train_dataloader=train_loader,
                    val_dataloaders=[valid_loader])
        with open(log_file, "a") as f:
            print("[*] Finish training.", file=f)


    # Independent test phase (embedding extraction and fold prediction)
    elif args.phase == "test":

        # Create directories
        emb_dir = args.model_dir + "/pfr_scores"
        pred_dir = args.model_dir + "/dfc_predictions"
        create_dirs([emb_dir, pred_dir])
        # Define output files
        test_name = os.path.basename(args.test_file).rsplit(".", 1)[0]
        emb_file = emb_dir + "/" + test_name + ".pkl"
        pred_file = pred_dir + "/" + test_name + ".pkl"
        pred_logits_file = pred_dir + "/" + test_name + "_logits.pkl"
        # Load the Lightning model for embedding extraction
        model = MultiClassTrain.load_from_checkpoint(
            args.model_file, net=net, hparams=args, log_file=log_file
        )
        # Define trainer (with no logger or checkpoints) with `TestSerializer`
        # callback for saving the embeddings and predictions
        trainer = pl.Trainer(
            gpus=1, logger=False, checkpoint_callback=False,
            callbacks=[TestSerializer(emb_path=emb_file,
                                      pred_path=pred_file,
                                      pred_logits_path=pred_logits_file)]
        )
        # Data loading
        with open(log_file, "a") as f:
            print("\n[*] Loading test data: %s" % test_name, file=f)
        test_loader = get_test_loader(args)
        # Test (supervised embedding and prediction extractor)
        trainer.test(model, test_dataloaders=test_loader)
        with open(log_file, "a") as f:
            print("[*] Embeddings saved in: %s" % emb_file, file=f)
            print("[*] Predictions saved in: %s" % pred_file, file=f)
            print("[*] Pred logits saved in: %s" % pred_logits_file, file=f)


if __name__ == "__main__":
    args = create_parser_args()
    main(args)
