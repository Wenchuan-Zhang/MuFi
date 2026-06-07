#!/bin/bash
# ============================================================================
# MuFi: Attention-based multimodal fusion transformer for predicting the
# efficacy of neoadjuvant therapy in breast cancer.
# Trains the MuFi model that fuses three-level pathomics tokens (cell / tissue
# / global WSI) with radiomics (pathway) tokens via a multimodal transformer.
# ============================================================================

# ---- Paths (edit these) ----------------------------------------------------
BASE_DIR="."                                   # repo root (where main.py lives)
DATA_ROOT_CLS256="/path/to/features/cls256"    # cell-level WSI features  (224/256 patches, e.g. H-optimus)
DATA_ROOT_CLS4096="/path/to/features/cls4096"  # tissue-level WSI features (4096 patches, e.g. HIPT4096)
DATA_ROOT_ONESLIDE="/path/to/features/oneslide"# global WSI-level features (e.g. Prov-Gigapath)
OMICS_DIR="datasets_csv/radiomics"             # radiomics / pathway token csv per case
LABEL_FILE="datasets_csv/metadata/mufi.csv"    # metadata csv with the response label
SPLIT_DIR="splits"                             # folder containing splits_{k}.csv
RESULTS_DIR="results_MuFi"                      # output directory

# ---- Task / model ----------------------------------------------------------
TASK="response"                                # binary pCR vs non-pCR
MODEL="survpath"                               # MuFi model dispatch key
TYPE_OF_PATH="combine"
LABEL_COL="response_lable"                      # response label column in LABEL_FILE
N_CLASSES=2

# ---- Hyperparameters (from the paper) --------------------------------------
LR=0.0005          # 5e-4
DECAY=0.0001       # weight decay 1e-4
OPT="adamW"
BATCH_SIZE=1
MAX_EPOCHS=30
NUM_PATCHES=4096
WSI_PROJ_DIM=256
ENCODING_DIM=1024
K=5                # 5-fold cross-validation

CUDA_VISIBLE_DEVICES=0 python main.py \
    --study mufi_breast --task $TASK --modality $MODEL \
    --split_dir $SPLIT_DIR --which_splits 5foldcv --k $K \
    --type_of_path $TYPE_OF_PATH \
    --data_root_dir_cls256 $DATA_ROOT_CLS256 \
    --data_root_dir_cls4096 $DATA_ROOT_CLS4096 \
    --data_root_dir_clsoneslide $DATA_ROOT_ONESLIDE \
    --omics_dir $OMICS_DIR \
    --label_file $LABEL_FILE --label_col $LABEL_COL \
    --results_dir $RESULTS_DIR \
    --bag_loss ce --n_classes $N_CLASSES \
    --batch_size $BATCH_SIZE --lr $LR --opt $OPT --reg $DECAY \
    --max_epochs $MAX_EPOCHS --num_patches $NUM_PATCHES \
    --wsi_projection_dim $WSI_PROJ_DIM --encoding_dim $ENCODING_DIM \
    --weighted_sample
