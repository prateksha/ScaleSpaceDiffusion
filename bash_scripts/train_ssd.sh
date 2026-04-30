#!/bin/bash

DATA_DIR="/path/to/images/"
DATASET_SPLIT_FILE="/path/to/train_split.txt"
RUN_ROOT="/path/to/experiment_runs/"

NUM_GPUS=2
NUM_WORKERS=4
DATASET_NAME="celeba"
IMAGE_SIZE=64
NUM_LEVELS=4

PREDICT_XSTART=true
SSD_CONFIG_FLAG=true
EXPERIMENT_NAME="ssd_${DATASET_NAME}_${IMAGE_SIZE}px_${NUM_LEVELS}L"


EXPERIMENT_DIR="${RUN_ROOT}/${EXPERIMENT_NAME}"
MODEL_DIR="${EXPERIMENT_DIR}/models"
mkdir -p "${MODEL_DIR}"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  scripts/image_train.py \
  experiment.name="${EXPERIMENT_NAME}" \
  experiment.experiment_type=ssd \
  experiment.root_dir="${EXPERIMENT_DIR}" \
  experiment.models_dir="${MODEL_DIR}" \
  experiment.num_workers="${NUM_WORKERS}" \
  dataset.name="${DATASET_NAME}" \
  dataset.data_dir="${DATA_DIR}" \
  dataset.dataset_split_file="${DATASET_SPLIT_FILE}" \
  model.image_size="${IMAGE_SIZE}" \
  diffusion.predict_xstart="${PREDICT_XSTART}" \
  ssd.ssd_config_flag="${SSD_CONFIG_FLAG}" \
  ssd.num_levels="${NUM_LEVELS}" \
