# FoldEmbeddings

This repository contains the source code to reproduce the results of the paper "An Analysis of Protein Language Model Embeddings for Fold Prediction" (see citation below).

## Downloadable data

Input data, protein-LM embeddings, and trained models can be found at <http://sigmat.ugr.es/~amelia/FoldEmbeddings/>.

## Run

### Input protein-LM embeddings

Use the code in the following sources to extract amino acid-level and protein-level LM embeddings (*LMEmb*):

- `UniRep` [1]: <https://github.com/songlab-cal/tape>
- `SeqVec` [2]: <https://github.com/mheinzinger/SeqVec>
- `ESM-1b` [3] and `ESM-MSA` [4]: <https://github.com/facebookresearch/esm>
- `ProtBERT` and `ProtT5` [5]: <https://github.com/agemagician/ProtTrans>

### Neural network models

```bash
./Run_train_test.sh
```

1. Train the `MLP`, `RBG`, and `LAT` models on `ProtT5` *LMEmb* embeddings using LMCL loss:

```bash
EMBTYPE="prot-t5"; INDIM=1024;
ARGSTRAIN="--phase=train --train_file=data/train/train.list
           --fold_label_file=data/train/fold_label_relation_1154.txt
           --num_classes=1154 --feats_dir=embeddings/${EMBTYPE}/train
           --input_dim=${INDIM} --loss_type=lmcl --loss_scale=30
           --batch_size_class=64 --ndata_workers=2"

# MLP (Multi-Layer Perceptron)
MODELDIR="models/${EMBTYPE}/MLP"; mkdir -p $MODELDIR
python scripts/main_lightning.py ${ARGSTRAIN} --model_dir=${MODELDIR} \
    --model_type="mlp" --loss_margin=0.2 \
    --hidden_dims="1024_512" --drop_prob=0.5 --batch_norm=True

# RBG (ResCNN-BGRU)
MODELDIR="models/${EMBTYPE}/RBG"; mkdir -p $MODELDIR
python scripts/main_lightning.py ${ARGSTRAIN} --model_dir=${MODELDIR} \
    --model_type="rescnn_gru" --loss_margin=0.6 \
    --channel_dims="512_${INDIM}_512_${INDIM}" --kernel_sizes="5_5" \
    --gru_dim=1024 --gru_bidirec=True --hidden_dims="512" \
    --activation_last="tanh" --drop_prob=0.2 --batch_norm=False

# LAT (Light-Attention)
MODELDIR="models/${EMBTYPE}/RBG"; mkdir -p $MODELDIR
python scripts/main_lightning.py ${ARGSTRAIN} --model_dir=${MODELDIR} \
    --model_type="light_att" --loss_margin=0.6 \
    --channel_dims="${INDIM}" --kernel_sizes="9" --hidden_dims="512" \
    --activation_last="tanh" --drop_prob=0.2 --batch_norm=False
```

2. Test the `RBG` model trained on `ProtT5` *LMEmb* embeddings:

    a. Extract *FoldEmb* embeddings and predictions for the LINDAHL and LINDAHL_1.75 test sets:

    ```bash
    EMBTYPE="prot-t5"; INDIM=1024;
    ARGSTEST="--phase=test --num_classes=1154 --loss_type=lmcl
              --model_type=rescnn_gru --input_dim=${INDIM}
              --channel_dims=512_${INDIM}_512_${INDIM} --kernel_sizes=5_5
              --gru_dim=1024 --gru_bidirec=True --hidden_dims=512
              --activation_last=tanh --drop_prob=0.2 --batch_norm=False
              --ndata_workers=2"

    # LINDAHL test set
    MODELDIR="models/${EMBTYPE}/RBG"
    python scripts/main_lightning.py ${ARGSTEST} --model_dir=${MODELDIR} \
        --model_file="${MODELDIR}/checkpoint/model_epoch80.ckpt" \
        --test_file="data/lindahl/lindahl.list" --scop_separation="_" \
        --feats_dir_test="embeddings/${EMBTYPE}/lindahl"

    # LINDAHL_1.75 test set
    python scripts/main_lightning.py ${ARGSTEST} --model_dir=${MODELDIR} \
        --model_file="${MODELDIR}/checkpoint/model_epoch80.ckpt" \
        --test_file="data/lindahl_1.75/lindahl_1.75.list" --scop_separation="." \
        --feats_dir_test="embeddings/${EMBTYPE}/lindahl_1.75"
    ```

    b. Pairwise Fold Recognition (`PFR`) task. Compute cosine similarity scores and evaluate (LINDAHL test set):

    ```bash
    SCORESDIR="models/prot-t5/RBG/pfr_scores"
    EMBEDFILE="${SCORESDIR}/lindahl.pkl"

    ./Run_eval_pairs.sh "lindahl" ${EMBEDFILE} ${SCORESDIR} "cosine"
    ```

    c. Direct Fold Classification (`DFC`) task. Classify and evaluate predictions (LINDAHL_1.75 test set):

    ```bash
    FOLDLABELFILE="data/train/fold_label_relation_1154.txt"
    PREDSDIR="models/prot-t5/RBG/dfc_predictions"
    PREDSFILE="${PREDSDIR}/lindahl_1.75.pkl"

    python scripts/eval/classify.py "lindahl_1.75" ${FOLDLABELFILE} ${PREDSFILE} \
        > "${PREDSDIR}/lindahl_1.75_results.txt"
    ```

### Ensemble models

- [`ESM-1b`, `ESM-MSA`, `ProtT5`] embeddings + [`RBG`, `LAT`] models.
- `PFR` task - Average and random forest ensemble (using cosine similarity scores).
- `DFC` task - Soft voting ensemble (using predicted logits).

```bash
MODELS="esm-1b-rbg+esm-1b-lat+esm-msa-rbg+esm-msa-lat+prot-t5-rbg+prot-t5-lat"
SCORESDIR="results/pfr_scores"
PREDSDIR="results/dfc_predictions"

# PFR - Average ensemble
python scripts/eval/average_ensembling.py "lindahl" ${SCORESDIR} ${MODELS}

# PFR - Random forest ensemble (+ 84 similarity measures)
./Run_random_forest.sh ${SCORESDIR} "sim+${MODELS}" 4

# DFC - Soft voting ensemble
python scripts/eval/soft_voting_ensembling.py "lindahl" ${PREDSDIR} ${MODELS}
```

## Requirements

- Python 3.7.7
- Numpy 1.19.0
- Scikit-Learn 0.23.1
- Matplotlib 3.2.2
- PyTorch 1.4.0
- Tensorboard 2.2.0
- PyTorch-Lightning 0.10.0

## Citation

A. Villegas-Morcillo, A.M. Gomez, and V. Sanchez, "An Analysis of Protein Language Model Embeddings for Fold Prediction," *Briefings in Bioinformatics*, bbac142 (2022). <https://doi.org/10.1093/bib/bbac142>

BibTex:

```bibtex
@article{villegas2022fold,
  author = {Villegas-Morcillo, Amelia and Gomez, Angel M. and Sanchez, Victoria},
  title = {An Analysis of Protein Language Model Embeddings for Fold Prediction},
  journal = {Briefings in Bioinformatics},
  year = {2022},
  month = {04},
  note = {bbac142},
  doi = {10.1093/bib/bbac142}
}
```

## References

[1] E.C. Alley et al., "Unified rational protein engineering with sequence-based deep representation learning," *Nature Methods* **16**, 12, pages 1315–1322 (2019). <https://doi.org/10.1038/s41592-019-0598-1>

[2] M. Heinzinger et al., "Modeling aspects of the language of life through transfer-learning protein sequences," *BMC Bioinformatics* **20**, 173 (2019). <https://doi.org/10.1186/s12859-019-3220-8>

[3] A. Rives et al., "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences", *Proceedings of the National Academy of Sciences* **118**, 15 (2021). <https://doi.org/10.1073/pnas.2016239118>

[4] R.M. Rao et al., "MSA transformer", *Proceedings of the 38th International Conference on Machine Learning (ICLR)*, PMLR **139**, pages 8844–8856 (2021). <https://proceedings.mlr.press/v139/rao21a.html>

[5] A. Elnaggar et al., "ProtTrans: Towards Cracking the Language of Lifes Code Through Self-Supervised Deep Learning and High Performance Computing," *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2021). <https://doi.org/10.1109/TPAMI.2021.3095381>
