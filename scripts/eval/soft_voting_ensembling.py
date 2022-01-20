#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: soft_voting_ensembling.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import pickle
import os
import sys
import numpy as np
from collections import defaultdict


def main(test_set, predictions_dir, model_subdirs):
    # read predictions and add logits
    model_subdirs = list(map(str, model_subdirs.split("+")))
    d = defaultdict(int)
    for mdl_dir in model_subdirs:
        pred_file =  "%s/%s/%s_logits.pkl" % (predictions_dir, mdl_dir, test_set)
        with open(pred_file, "rb") as f:
            d_single = pickle.load(f)
        for key in d_single.keys():
            d[key] += d_single[key]

    # create top 5 predictions file and save
    output_dict = {n: np.argsort(-d[n])[:5] for n in d.keys()}
    output_dir = predictions_dir + "/soft_voting_ensemble"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + "/%s.pkl" % test_set
    with open(output_file, "wb") as f:
        pickle.dump(output_dict, f)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: %s <test_set> <predictions_dir> <model_subdirs>" % sys.argv[0])
    test_set, predictions_dir, model_subdirs = sys.argv[1:]
    main(test_set, predictions_dir, model_subdirs)
