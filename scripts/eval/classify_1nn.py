#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: classify_1nn.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import pickle
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_feats_labels(list_file, fold_dict, emb_file):
    # read names and folds
    data = np.loadtxt(list_file, dtype=str)
    names = data[:,0]
    folds = data[:,3]
    # convert folds to labels
    y = np.array([fold_dict[item] for item in folds])
    # load embeddings
    embeddings = pickle.load(open(emb_file, "rb"))
    X = np.array([embeddings[item] for item in names])
    return X, y


def main(list_train, list_test, fold_label_file, embed_train, embed_test):

    # create dict for fold-label relation
    fold_label = np.loadtxt(fold_label_file, dtype=str)[1:]
    fold_dict = {item[0]: int(item[1]) for item in fold_label}

    # load training data
    X_train, y_train = load_feats_labels(list_train, fold_dict, embed_train)

    headers = ["Full set", "Family subset", "Superfamily subset", "Fold subset"]
    for suffix, head in zip(["", "_family", "_superfamily", "_fold"], headers):
        # load test data
        list_test_level = list_test + suffix
        X_test, y_test = load_feats_labels(list_test_level, fold_dict, embed_test)
        num = len(y_test)
        # compute cosine matrix
        cosine_mat = cosine_similarity(X_test, X_train)
        pos = np.argsort(-cosine_mat, axis=1)
        # calculate top1/top5 prediction accuracy
        correct_top1 = np.sum(np.equal(y_train[pos[:,0]], y_test))
        acc_top1 = correct_top1 / num * 100
        correct_top5 = np.sum(
            np.equal(y_train[pos[:,:5]], y_test.reshape(-1,1)).any(axis=1)
        )
        acc_top5 = correct_top5 / num * 100
        print("[*] %s:" % head)
        print("Top1 accuracy: %.2f [%d/%d]" % (acc_top1, correct_top1, num))
        print("Top5 accuracy: %.2f [%d/%d]" % (acc_top5, correct_top5, num))


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s <list_train> <list_test> <fold_label_file> "
                 "<embed_train> <embed_test> " % sys.argv[0])
    list_train, list_test, fold_label_file, embed_train, embed_test = sys.argv[1:]
    main(list_train, list_test, fold_label_file, embed_train, embed_test)
