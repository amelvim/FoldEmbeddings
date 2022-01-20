#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: classify.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import sys
import pickle
import numpy as np


def main(list_file, fold_label_file, prediction_file):

    # create dict for fold-label relation
    fold_label = np.loadtxt(fold_label_file, dtype=str)[1:]
    fold_dict = {item[0]: int(item[1]) for item in fold_label}

    # load predictions dictionary
    with open(prediction_file, "rb") as h:
        preds_dict = pickle.load(h)

    headers = ["Full set", "Family subset", "Superfamily subset", "Fold subset"]
    for suffix, head in zip(["", "_family", "_superfamily", "_fold"], headers):
        # read names and folds
        list_test_level = list_file + suffix
        data = np.loadtxt(list_test_level, dtype=str)
        names = data[:,0]
        folds = data[:,3]
        # convert folds to labels
        labels = np.array([fold_dict[item] for item in folds])
        num = len(labels)
        # calculate top1/top5 prediction accuracy
        correct_top1 = np.sum(
            [preds_dict[n][0] == labels[i] for i, n in enumerate(names)]
        )
        acc_top1 = correct_top1 / num * 100
        correct_top5 = np.sum(
            [labels[i] in preds_dict[n] for i, n in enumerate(names)]
        )
        acc_top5 = correct_top5 / num * 100
        print("[*] %s:" % head)
        print("Top1 accuracy: %.2f [%d/%d]" % (acc_top1, correct_top1, num))
        print("Top5 accuracy: %.2f [%d/%d]" % (acc_top5, correct_top5, num))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: %s <list_file> <fold_label_file> <prediction_file>" \
                 % sys.argv[0])
    list_file, fold_label_file, prediction_file = sys.argv[1:]
    main(list_file, fold_label_file, prediction_file)
