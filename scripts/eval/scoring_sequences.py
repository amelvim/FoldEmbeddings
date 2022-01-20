#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: scoring_sequences.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import sys
import pickle
import numpy as np
from scipy.special import softmax
from sklearn.metrics import pairwise_distances


def calculate_similarity(feats1, feats2, dist="cosine"):
    ''' Calculate similarity distance between two sequences of feature vectors.
        SSA - Soft Symmetric Alignment (Bepler and Berger, 2019)
    '''
    distance_matrix = pairwise_distances(feats1, feats2, metric=dist)
    if dist == "cosine" or dist == "correlation":
        similarity_matrix = 1 - distance_matrix
    elif dist == "euclidean" or dist == "manhattan":
        similarity_matrix = -distance_matrix
    else:
        print("ERROR: Invalid distance measure --%s--, try --cosine--, "
              "--euclidean--, --manhattan-- or --correlation--" % dist)
        sys.exit(1)

    # soft symmetric alignment
    alpha = softmax(similarity_matrix, axis=1)
    beta = softmax(similarity_matrix, axis=0)
    a = alpha + beta - alpha * beta
    return np.sum(a * similarity_matrix) / np.sum(a)


def main(names_file, embedding_subdir, score_file, distance_measure):
    # read names
    names = np.loadtxt(names_file, dtype="str")
    N = len(names)

    # calculate similarity for each pair of sequences
    results = []
    for i in range(N):
        feats1 = np.load("%s/%s.npy" % (embedding_subdir, names[i]))
        for j in range(i+1, N):
            feats2 = np.load("%s/%s.npy" % (embedding_subdir, names[j]))
            sim = calculate_similarity(feats1, feats2, distance_measure)
            results.append([names[i], names[j], sim])
            results.append([names[j], names[i], sim])

    # sort pairs and save
    results.sort()
    np.savetxt(score_file, np.array(results), fmt="%s")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s <names_file> <embedding_subdir> <score_file> "
                 "<distance_measure>" % sys.argv[0])
    names_file, embedding_subdir, score_file, distance_measure = sys.argv[1:]
    main(names_file, embedding_subdir, score_file, distance_measure)
