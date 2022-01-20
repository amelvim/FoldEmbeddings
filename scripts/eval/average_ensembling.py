#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: average_ensembling.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import os
import sys
import numpy as np
from collections import defaultdict


def main(test_set, scores_dir, model_subdirs):
    # read scores and append to list
    model_subdirs = list(map(str, model_subdirs.split("+")))
    d = defaultdict(list)
    for mdl_dir in model_subdirs:
        score_file =  "%s/%s/%s.score" % (scores_dir, mdl_dir, test_set)
        pairs_scores = np.loadtxt(score_file, dtype=str)
        for x in pairs_scores:
            pair = " ".join(x[:2])
            score = float(x[2:])
            d[pair].append(score)

    # create pairs-scores file (mean) and save
    pairs_mean_scores = np.array(
        [key.split(" ") + [np.mean(d[key])] for key in d.keys()]
    )
    output_dir = scores_dir + "/average_ensemble"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(output_dir + "/%s.score" % test_set, pairs_mean_scores, fmt="%s")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: %s <test_set> <scores_dir> <model_subdirs>" % sys.argv[0])
    test_set, scores_dir, model_subdirs = sys.argv[1:]
    main(test_set, scores_dir, model_subdirs)
