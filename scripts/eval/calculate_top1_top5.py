#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: calculate_top1_top5.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import sys
import numpy as np


def get_top_indexs(scores):
    scores = np.array(scores)
    top_inds = np.argsort(scores)[::-1]
    # get top1
    top1 = list(top_inds[:1])
    for i in top_inds[1:]:
        if scores[i] == scores[top1[0]]:
            top1.append(i)
    # get top5
    top5 = list(top_inds[:5])
    for i in top_inds[5:]:
        if scores[i] == scores[top5[4]]:
            top5.append(i)
    assert len(top1) < 15 and len(top5) < 30
    return top1, top5


def main(score_file, level_pairs_file):
    # score dict
    pairs_scores = np.loadtxt(score_file, dtype=str)
    score_dict = {}
    for i, p in enumerate(pairs_scores):
        if p[0] not in score_dict:
            score_dict[p[0]] = []
        score_dict[p[0]].append(((p[0], p[1]), float(p[2])))

    # pairs data
    positive_pairs = np.loadtxt(level_pairs_file, dtype=str)
    single_names = np.unique(positive_pairs[:,0])
    num_samples = len(single_names)
    positive_pairs = list(map(tuple, positive_pairs)) # convert numpy matrix to list of tuples

    # calculte top1 top5
    top = [0, 0]
    for name in single_names:
        tmp_scores = [s[1] for s in score_dict[name]]
        top1, top5 = get_top_indexs(tmp_scores)
        for k in top1:
            if score_dict[name][k][0] in positive_pairs:
                top[0] += 1
                break
        for k in top5:
            if score_dict[name][k][0] in positive_pairs:
                top[1] += 1
                break

    print("Test_number:", num_samples)
    print("Top_number:", top)
    print("Sensitivity:", "%4.1f %4.1f" % tuple([i/num_samples*100 for i in top]))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: %s <score_file> <level_pairs_file>" % sys.argv[0])
    score_file, level_pairs_file = sys.argv[1:]
    main(score_file, level_pairs_file)
