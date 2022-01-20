#!/bin/bash
###
# File: Run_train_test.sh
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


###############################################################################
# Define arguments
DATASET=""
# "" / deepsf
PHASE="train"
# train / test
EMBTYPE="prot-t5"
# unirep / seqvec / esm-1b / esm-msa / prot-bert / prot-t5
NET="rbg"
# mlp / rbg / lat
ACTIV="tanh"
# sigmoid / tanh
LOSSTYPE="lmcl"
# softmax / lmcl
SCALE=30
NWORKERS=1
###############################################################################

# Specify files and options for each dataset
if [[ $DATASET != "deepsf" ]]; then
    TRAINFILE="data/train/train.list"
    FOLDLABELFILE="data/train/fold_label_relation_1154.txt"
    EMBDIRTRAIN="embeddings/${EMBTYPE}/train"
    MODELPREFIX="models"; NCLASSES=1154;
    TESTARRAY=("lindahl" "lindahl_1.75")

else
    TRAINFILE="data/deepsf_train/deepsf_train.list"
    FOLDLABELFILE="data/deepsf_train/fold_label_relation_1195.txt"
    EMBDIRTRAIN="embeddings/${EMBTYPE}/deepsf_train"
    MODELPREFIX="models_deepsf"; NCLASSES=1195;
    TESTARRAY=("deepsf_scop_2.06")
fi

# Set arguments for embedding type
if [[ $EMBTYPE == "unirep" ]]; then INDIM=1900;
elif [[ $EMBTYPE == "seqvec" ]]; then INDIM=1024;
elif [[ $EMBTYPE == "esm-1b" ]]; then INDIM=1280;
elif [[ $EMBTYPE == "esm-msa" ]]; then INDIM=768;
elif [[ $EMBTYPE == "prot-bert" || $EMBTYPE == "prot-t5" ]]; then INDIM=1024;
fi

# Set arguments for model type
if [[ $NET == "mlp" ]]; then
    MODELOPTS="--model_type=mlp --hidden_dims=1024_512 --drop_prob=0.5
               --batch_norm=True --loss_margin=0.2"

elif [[ $NET == "rbg" ]]; then
    MODELOPTS="--model_type=rescnn_gru --channel_dims=512_${INDIM}_512_${INDIM}
               --kernel_sizes=5_5 --gru_dim=1024 --gru_bidirec=True
               --hidden_dims=512 --drop_prob=0.2 --batch_norm=False
               --activation_last=${ACTIV} --loss_margin=0.6"

elif [[ $NET == "lat" ]]; then
    MODELOPTS="--model_type=light_att --channel_dims=${INDIM}
               --kernel_sizes=9 --hidden_dims=512 --drop_prob=0.2
               --batch_norm=False --activation_last=${ACTIV} --loss_margin=0.6"
fi

MODELDIR=${MODELPREFIX}/${EMBTYPE}/${NET^^}
mkdir -p ${MODELDIR}


###############################################################################
# Training phase
if [[ $PHASE == "train" ]]; then

echo "[*] Training phase..."
python scripts/main_lightning.py --phase="train" \
    --train_file=${TRAINFILE} --fold_label_file=${FOLDLABELFILE} \
    --feats_dir=${EMBDIRTRAIN} --model_dir=${MODELDIR} \
    --loss_type=${LOSSTYPE} --loss_scale=${SCALE} \
    --input_dim=${INDIM} --num_classes=${NCLASSES} ${MODELOPTS} \
    --batch_size_class=64 --ndata_workers=${NWORKERS}


###############################################################################
# Test phase
elif [[ $PHASE == "test" ]]; then

CKPTFILE="${MODELDIR}/checkpoint/model_epoch80.ckpt"

for TESTSET in ${TESTARRAY[@]}; do
    # Extract embeddings and predictions
    if [[ $TESTSET == "lindahl" ]]; then SEP="_"; else SEP="."; fi
    TESTFILE="data/${TESTSET}/${TESTSET}.list"
    EMBDIRTEST="embeddings/${EMBTYPE}/${TESTSET}"

    echo "[*] Extracting ${TESTSET^^} embeddings and predictions..."
    python scripts/main_lightning.py --phase="test" \
        --test_file=${TESTFILE} --scop_separation=${SEP} \
        --feats_dir_test=${EMBDIRTEST} --model_dir=${MODELDIR} \
        --model_file=${CKPTFILE} --loss_type=${LOSSTYPE} \
        --input_dim=${INDIM} --num_classes=${NCLASSES} ${MODELOPTS} \
        --ndata_workers=${NWORKERS}

    if [[ $TESTSET == "lindahl" ]]; then
        # Compute cosine similarity scores and evaluate
        SCORESDIR="${MODELDIR}/pfr_scores"
        EMBEDFILE="${SCORESDIR}/${TESTSET}.pkl"

        echo "[*] Computing cosine similarity scores and evaluating..."
        ./Run_eval_pairs.sh ${TESTSET} ${EMBEDFILE} ${SCORESDIR} "cosine"

    elif [[ $TESTSET == "lindahl_1.75" || $TESTSET == "deepsf_scop_2.06" ]]; then
        # Classify and evaluate predictions
        PREDSDIR="${MODELDIR}/dfc_predictions"
        PREDSFILE="${PREDSDIR}/${TESTSET}.pkl"

        echo "[*] Classifying and evaluating predictions..."
        python scripts/eval/classify.py ${TESTFILE} ${FOLDLABELFILE} ${PREDSFILE} \
            > "${PREDSDIR}/${TESTSET}_results.txt"
    fi
done

fi
