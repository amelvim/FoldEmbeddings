#!/bin/bash
###
# File: Run_eval_pairs_sequences.sh
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


DATASET=$1
EMBEDDIR=$2
OUTDIR=$3
DISTANCE=$4
# cosine / manhattan

# Get scoring for each pair of proteins
NAMESFIL=data/${DATASET}/level_pairs/${DATASET}_names
SCOREFIL=${OUTDIR}/${DATASET}.score
mkdir -p ${OUTDIR}

EMBEDSUBDIR=${EMBEDDIR}/${DATASET}
python scripts/eval/scoring_sequences.py ${NAMESFIL} ${EMBEDSUBDIR} ${SCOREFIL} ${DISTANCE}

FAMFIL=data/${DATASET}/level_pairs/${DATASET}_family
SUPFAMFIL=data/${DATASET}/level_pairs/${DATASET}_superfamily
FOLDFIL=data/${DATASET}/level_pairs/${DATASET}_fold

# Get Top1 / Top5 accuracy predictions
RESFIL=${OUTDIR}/${DATASET}_results.txt
TMPDIR=${OUTDIR}/TMP
mkdir -p ${TMPDIR}

printf "Calculating correctly predicted template at family level (Top1, Top5): \n" > ${RESFIL}
python scripts/eval/calculate_top1_top5.py ${SCOREFIL} ${FAMFIL} >> ${RESFIL}
printf "\nCalculating correctly predicted template at superfamily level (Top1, Top5): \n" >> ${RESFIL}
grep -F -v -f  ${FAMFIL} ${SCOREFIL} > ${TMPDIR}/deleted-fam
python scripts/eval/calculate_top1_top5.py ${TMPDIR}/deleted-fam ${SUPFAMFIL} >> ${RESFIL}
printf "\nCalculating correctly predicted template at fold level (Top1, Top5): \n" >> ${RESFIL}
grep -F -v -f ${SUPFAMFIL} ${TMPDIR}/deleted-fam > ${TMPDIR}/deleted-fam-supfam
python scripts/eval/calculate_top1_top5.py ${TMPDIR}/deleted-fam-supfam ${FOLDFIL} >> ${RESFIL}
printf "" >> ${RESFIL}

rm -r $TMPDIR
