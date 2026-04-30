#!/bin/bash

RUN_ROOT="/path/to/experiment_runs/"
MODEL_PATH="/path/to/downloaded/checkpoint.pt"

DATASET_NAME="celeba"
IMAGE_SIZE=64
NUM_LEVELS=4
NUM_SAMPLES=500

EXPERIMENT_NAME="ssd_${DATASET_NAME}_${IMAGE_SIZE}px_${NUM_LEVELS}L"
EXPERIMENT_DIR="${RUN_ROOT}/${EXPERIMENT_NAME}"
INFERENCE_DIR="${EXPERIMENT_DIR}/inferencing/"

NUM_GPUS=1 
SSD_CONFIG_FLAG=true
PREDICT_XSTART=true
CHUNKING_FLAG=false


if [ ! -f "${MODEL_PATH}" ]; then
  echo "Missing checkpoint: ${MODEL_PATH}"
  exit 1
fi

mkdir -p "${INFERENCE_DIR}"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  scripts/image_sample.py \
  hydra.run.dir="${INFERENCE_DIR}" \
  experiment.experiment_type=ssd \
  experiment.root_dir="${EXPERIMENT_DIR}" \
  experiment.name="${EXPERIMENT_NAME}" \
  model.image_size="${IMAGE_SIZE}" \
  diffusion.predict_xstart="${PREDICT_XSTART}" \
  ssd.ssd_config_flag="${SSD_CONFIG_FLAG}" \
  ssd.num_levels="${NUM_LEVELS}" \
  inference.inferencing_flag=true \
  inference.num_samples="${NUM_SAMPLES}" \
  inference.chunking.enabled="${CHUNKING_FLAG}" \
  inference.model_path="${MODEL_PATH}" \
