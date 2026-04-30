import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
"""
Train a diffusion model on images.
"""
import sys
import os

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import create_model_and_diffusion
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.ssd_utils import get_resolutions_array
from guided_diffusion.global_config import DEFAULT_CHANNEL_MULTS
from torchinfo import summary
import torch
import pprint

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    if cfg.model.channel_mult == "":
        if cfg.model.image_size in DEFAULT_CHANNEL_MULTS:
            channel_mult = DEFAULT_CHANNEL_MULTS[cfg.model.image_size]
        else:
            raise ValueError(f"Unsupported image size: {cfg.model.image_size}")
        cfg.model.channel_mult = channel_mult
    if cfg.ssd.ssd_config_flag:
        ssd_config = OmegaConf.create(cfg.ssd)   
        ssd_config.scale_max = cfg.model.image_size
        ssd_config.model_channel_mult = cfg.model.channel_mult
        resolutions_array, unet_res_layer_map = get_resolutions_array(ssd_config)

        with open_dict(cfg.ssd):
            # Add new keys that didn't exist in YAML
            cfg.ssd.scale_min = resolutions_array[0]
            cfg.ssd.unet_res_layer_map = unet_res_layer_map
            
            cfg.ssd.scale_max = cfg.model.image_size
            cfg.ssd.model_channel_mult = cfg.model.channel_mult

    else:
        resolutions_array = None
        unet_res_layer_map = None

    dist_util.setup_dist()

    resume_step = 0
    latest_model_paths = [
        os.path.join(cfg.experiment.models_dir, f"model_latest_{i}.pt") for i in (0, 1)
    ]

    for path in latest_model_paths:
        if os.path.exists(path):
            try:
                checkpoint = dist_util.load_state_dict(path, map_location=dist_util.dev())
                step_read = checkpoint.get("step", 0)
                if step_read > resume_step:
                    resume_step = step_read
                logger.log(f"Loaded model from checkpoint: {path}")
                break
            except Exception:
                continue  # try next checkpoint if this fails
    logger.configure(dir=cfg.experiment.models_dir, format_strs=["stdout", "csv", "tensorboard", "log"], resume_step=resume_step // cfg.training.log_interval)

    logger.log(f"[Device: {dist_util.dev()}] logging config; resume_step={resume_step}")
    logger.log(f"[Device: {dist_util.dev()}] {pprint.pformat(OmegaConf.to_container(cfg, resolve=True))}")
    logger.log(f"[Device: {dist_util.dev()}] resolutions_array={resolutions_array} and unet_res_layer_map={unet_res_layer_map}")
    logger.log(f"[Device: {dist_util.dev()}] creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        cfg,
        resolutions_array=resolutions_array,
        unet_res_layer_map=unet_res_layer_map
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(cfg.experiment.schedule_sampler, diffusion)

    logger.log(f"[Device: {dist_util.dev()}] creating data loader...")
    data = load_data(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.training.batch_size,
        image_size=cfg.model.image_size,
        class_cond=cfg.model.class_cond,
        dataset_split_file=cfg.dataset.dataset_split_file,
        num_workers=cfg.experiment.num_workers,
    )

    logger.log(f"[Device: {dist_util.dev()}] training...")

    training_args = OmegaConf.to_container(cfg.training, resolve=True)
    training_args.update({
        "model": model,
        "diffusion": diffusion,
        "data": data,
        "resume_checkpoint": cfg.experiment.resume_checkpoint,
        "schedule_sampler": schedule_sampler,
        "models_dir": cfg.experiment.models_dir,
        "ssd_config": getattr(cfg, "ssd", None), # Safe access in case ssd is missing
    })

    TrainLoop(**training_args).run_loop()

if __name__ == "__main__":
    main()
