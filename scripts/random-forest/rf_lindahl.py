#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: rf_lindahl.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def get_labels_scores(pairs_file, scores_files):
    # Load pairs and labels
    pairs_labels = np.loadtxt(pairs_file, dtype=str)
    featsdict = {x[0] + " " + x[1]: np.array(x[2:], dtype=np.float) \
                 for x in pairs_labels}
    # Load scores
    for scfile in scores_files:
        pairs_scores = np.loadtxt(scfile, dtype=str)
        for x in pairs_scores:
            pair = x[0] + " " + x[1]
            scores = np.array(x[2:], dtype=np.float)
            featsdict[pair] = np.append(featsdict[pair], scores)
    return featsdict


def main(pairs_file, scores_files, train_file, test_file, save_file, njobs):

    print("[*] Loading all data (Lindahl dataset)")
    scores_files = list(map(str, scores_files.split("+")))
    featsdict = get_labels_scores(pairs_file, scores_files)

    print("[*] Separating data in train-test")
    with open(train_file, "r") as f:
        train_pairs = f.read().splitlines()
    with open(test_file, "r") as f:
        test_pairs = f.read().splitlines()

    feats_train = np.array([featsdict[item] for item in train_pairs])
    X_train = feats_train[:, 1:]
    y_train = feats_train[:, :1].squeeze()
    print("[*] Train shape:", X_train.shape)

    feats_test = np.array([featsdict[item] for item in test_pairs])
    X_test = feats_test[:, 1:]
    y_test = feats_test[:, :1].squeeze()
    print("[*] Test shape:", X_test.shape)

    clf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=njobs)
    print("[*] Training...")
    clf.fit(X_train, y_train)
    print("[*] Testing...")
    y_prob = clf.predict_proba(X_test)

    print("[*] Saving probability results")
    ids = np.array(test_pairs)
    output = np.vstack((ids, y_prob[:,1])).T
    np.savetxt(save_file, output, delimiter=" ", fmt="%s")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit("Usage: %s <pairs_file> <scores_files> <train_file> "
                 "<test_file> <save_file> <njobs>" % sys.argv[0])
    pairs_file, scores_files, train_file, test_file, save_file, njobs = sys.argv[1:]
    main(pairs_file, scores_files, train_file, test_file, save_file, int(njobs))
