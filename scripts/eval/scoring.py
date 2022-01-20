#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: scoring.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import sys
import pickle
import numpy as np
from scipy.spatial import distance


def calculate_similarity(feat1, feat2, dist="cosine"):
    ''' Calculate similarity distance between two feature vectors '''
    if dist == "cosine":
        return 1 - distance.cosine(feat1, feat2)
    elif dist == "euclidean":
        return -distance.euclidean(feat1, feat2)
    elif dist == "manhattan":
        return -distance.cityblock(feat1, feat2)
    elif dist == "correlation":
        return 1 - distance.correlation(feat1, feat2)
    else:
        print("ERROR: Invalid distance measure --%s--, try --cosine--, "
              "--euclidean--, --manhattan-- or --correlation--" % dist)
        sys.exit(1)


def main(names_file, embedding_file, score_file, distance_measure):
    # read names
    names = np.loadtxt(names_file, dtype="str")

    # parse pairs
    pairs = [(i, j) for i in names for j in names if i != j]

    # load features dictionary
    with open(embedding_file, "rb") as h:
        featdict = pickle.load(h)

    # calculate similarity for each pair
    with open(score_file, "w") as fout:
        for i, p in enumerate(pairs):
            sim = calculate_similarity(featdict[p[0]], featdict[p[1]],
                                       distance_measure)
            print(p[0], p[1], sim, file=fout)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s <names_file> <embedding_file> <score_file> "
                 "<distance_measure>" % sys.argv[0])
    names_file, embedding_file, score_file, distance_measure = sys.argv[1:]
    main(names_file, embedding_file, score_file, distance_measure)
