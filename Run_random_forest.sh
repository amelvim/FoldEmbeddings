#!/bin/bash
###
# File: Run_random_forest.sh
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


DATASET="lindahl"
SCORESDIR=$1
MODELS=$2
MDLARR=(`echo $MODELS | tr "+" " "`)
# sim+esm-msa-rbg+prot-t5-rbg -> (sim esm-msa-rbg prot-t5-rbg)
NJOBS=$3

OUTDIR="${SCORESDIR}/random_forest_ensemble"
mkdir -p ${OUTDIR}

SCORESFILES="${SCORESDIR}/${MDLARR[0]}/${DATASET}.score"
for MDL in ${MDLARR[@]:1}; do
    SCORESFILES+="+${SCORESDIR}/${MDL}/${DATASET}.score"
done

PAIRSFILE="data/${DATASET}/${DATASET}.pairs"

for PART in {1..10}; do
    TRAINFILE="data/${DATASET}/cv10_pairs/train${PART}.pairs"
    TESTFILE="data/${DATASET}/cv10_pairs/test${PART}.pairs"
    SAVEFILE="${OUTDIR}/${DATASET}_${PART}.score"
    python scripts/random-forest/rf_lindahl.py \
        $PAIRSFILE $SCORESFILES $TRAINFILE $TESTFILE $SAVEFILE $NJOBS
done
cat ${OUTDIR}/${DATASET}_{1..10}.score > ${OUTDIR}/${DATASET}.score
