# Scale Space Diffusion

This is the codebase for [Scale Space Diffusion](https://arxiv.org/abs/2603.08709) (Accepted at CVPR 2026).

This repository is built on top of [openai/guided-diffusion](https://github.com/openai/guided-diffusion)

# Download pre-trained models

We have released checkpoints for the main models in the paper. 

Here are the download links for each model checkpoint:

CelebA-64
 * SSD (Flexi-UNet, 2L, 1M checkpoint): [ssd_64res_celeba_2levels_ema_0.9999_1000000.pt](https://huggingface.co/prateksha-u/ssd/resolve/main/ssd_64res_celeba_2levels_ema_0.9999_1000000.pt)
 * SSD (Flexi-UNet, 4L, 1M checkpoint): [ssd_64res_celeba_4levels_ema_0.9999_1000000.pt](https://huggingface.co/prateksha-u/ssd/resolve/main/ssd_64res_celeba_4levels_ema_0.9999_1000000.pt)

CelebA-128
 * SSD (Flexi-UNet, 3L, 300K checkpoint): [ssd_128res_celeba_3levels_ema_0.9999_300000.pt](https://huggingface.co/prateksha-u/ssd/resolve/main/ssd_128res_celeba_3levels_ema_0.9999_300000.pt)
 * SSD (Flexi-UNet, 5L, 300K checkpoint): [ssd_128res_celeba_5levels_ema_0.9999_300000.pt](https://huggingface.co/prateksha-u/ssd/resolve/main/ssd_128res_celeba_5levels_ema_0.9999_300000.pt)

CelebA-256
 * SSD (Flexi-UNet, 3L, 300K checkpoint): [ssd_256res_celeba_3levels_ema_0.9999_300000.pt](https://huggingface.co/prateksha-u/ssd/resolve/main/ssd_256res_celeba_3levels_ema_0.9999_300000.pt)
 * SSD (Flexi-UNet, 6L, 300K checkpoint): [ssd_256res_celeba_6levels_ema_0.9999_300000.pt](https://huggingface.co/prateksha-u/ssd/resolve/main/ssd_256res_celeba_6levels_ema_0.9999_300000.pt)

ImageNet-64
 * SSD (Flexi-UNet, 2L, 1M checkpoint): [ssd_64res_imagenet_2levels_ema_0.9999_1000000.pt](https://huggingface.co/prateksha-u/ssd/resolve/main/ssd_64res_imagenet_2levels_ema_0.9999_1000000.pt)

# Setting up the environment

Clone this repository and run the following commands to setup the codebase:

```bash
git clone https://github.com/prateksha/ScaleSpaceDiffusion.git
cd ScaleSpaceDiffusion

conda create -n ssd_env python=3.12 -y
conda activate ssd_env

pip install -e .
pip install pillow
pip install numpy
pip install pandas
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install tensorflow
pip install tensorboard
pip install torchinfo
pip install hydra-core --upgrade
```

# Sampling from pre-trained models

Use `bash_scripts/sample_from_pretrained_model.sh` to run inference from a downloaded checkpoint. Download one of the pretrained model files above, then edit the script to point to that checkpoint and choose where outputs should be written.

```bash
RUN_ROOT="/path/to/ssd_code_release/runs/"
EXPERIMENT_NAME="ssd_celeba_64px_4L"
MODEL_PATH="/path/to/downloaded/ssd_64res_celeba_4levels_ema_0.9999_1000000.pt"
```

Set `IMAGE_SIZE`, `NUM_LEVELS`, and `NUM_SAMPLES` in the same script. `IMAGE_SIZE` and `NUM_LEVELS` must match the downloaded checkpoint so that the correct model architecture is constructed.

```bash
IMAGE_SIZE=64
NUM_LEVELS=4
NUM_SAMPLES=500
```

Run inference with:

```bash
bash bash_scripts/sample_from_pretrained_model.sh
```

The script writes samples under `${RUN_ROOT}/${EXPERIMENT_NAME}/inferencing/`.

# Training models

Use `bash_scripts/train_ssd.sh` to train your own SSD model. Set `DATA_DIR` to the directory containing all images, and set `DATASET_SPLIT_FILE` to a text file containing the filenames of the images to use for training.

```bash
DATA_DIR="/path/to/images/"
DATASET_SPLIT_FILE="/path/to/train_split.txt"
RUN_ROOT="/path/to/experiment_runs/"
```

`DATASET_NAME` is used to create the experiment name. Set `IMAGE_SIZE` and `NUM_LEVELS` for the model you want to train:

```bash
DATASET_NAME="celeba"
IMAGE_SIZE=64
NUM_LEVELS=4
```

The script creates:

```bash
EXPERIMENT_NAME="ssd_${DATASET_NAME}_${IMAGE_SIZE}px_${NUM_LEVELS}L"
EXPERIMENT_DIR="${RUN_ROOT}/${EXPERIMENT_NAME}"
MODEL_DIR="${EXPERIMENT_DIR}/models"
```

Run training with:

```bash
bash bash_scripts/train_ssd.sh
```

By default, model checkpoints are dumped every 10K iterations. To change this, add or edit the Hydra override for `training.save_interval` in `bash_scripts/train_ssd.sh`, for example:

## Citation
```
@article{mukhopadhyay2026scale,
  title={Scale Space Diffusion},
  author={Mukhopadhyay, Soumik and Udhayanan, Prateksha and Shrivastava, Abhinav},
  journal={arXiv preprint arXiv:2603.08709},
  year={2026}
}
```
