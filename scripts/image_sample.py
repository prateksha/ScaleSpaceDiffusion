"""
Generate a large batch of image samples from a model using Hydra for configuration.
"""

import os
import time
import math
import torch as th
import torch.distributed as dist
from PIL import Image

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from guided_diffusion import dist_util, logger
from guided_diffusion.ssd_utils import get_resolutions_array
from guided_diffusion.global_config import DEFAULT_CHANNEL_MULTS
from guided_diffusion.script_util import (
    NUM_CLASSES,
    create_model_and_diffusion,
)

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
    checkpoint_state = dist_util.load_state_dict(cfg.inference.model_path, map_location="cpu")
    checkpoint_step = checkpoint_state.get("step", 0)
    inference_checkpoint = f"{int(checkpoint_step):06d}"
    if "ema" in os.path.basename(cfg.inference.model_path):
        ema_tag = "_".join(os.path.basename(cfg.inference.model_path).replace(".pt", "").split("_")[:2])
        inference_checkpoint = f"{inference_checkpoint}_{ema_tag}"
    with open_dict(cfg.inference):
        cfg.inference.inference_checkpoint = inference_checkpoint

    inferencing_dir = os.path.join(
        cfg.experiment.root_dir,
        "inferencing",
        cfg.inference.inference_checkpoint,
    )

    os.makedirs(inferencing_dir, exist_ok=True)
    logger.configure(dir=inferencing_dir)
    
    logger.log(f"--- Starting Inference ---")
    logger.log(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.log(f"Inferencing dir: {inferencing_dir}")

    # 4. Create Model
    model, diffusion = create_model_and_diffusion(
        cfg,
        resolutions_array=resolutions_array,
        unet_res_layer_map=unet_res_layer_map
    )

    # 5. Load Checkpoint
    ckpt_path = cfg.inference.model_path
    logger.log(f"Loading checkpoint: {ckpt_path}")

    model_state = checkpoint_state["model"] if "model" in checkpoint_state else checkpoint_state
    model.load_state_dict(model_state)
    model.to(dist_util.dev())
    
    if cfg.model.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # 6. Calculate Workload (DDP aware)
    # Split total samples across all ranks
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    total_samples = int(cfg.inference.num_samples)
    chunking_flag = bool(cfg.inference.chunking.enabled)
    if chunking_flag:
        num_chunks = int(cfg.inference.chunking.num_chunks)
        chunk_id = int(cfg.inference.chunking.chunk_id)
        samples_per_chunk = math.ceil(total_samples / num_chunks)
        chunk_start = min(chunk_id * samples_per_chunk, total_samples)
        chunk_end = min(chunk_start + samples_per_chunk, total_samples)
    else:
        chunk_start = 0
        chunk_end = total_samples

    chunk_samples = max(0, chunk_end - chunk_start)
    samples_per_rank = math.ceil(chunk_samples / world_size) if chunk_samples > 0 else 0
    rank_start = chunk_start + (rank * samples_per_rank)
    rank_end = min(rank_start + samples_per_rank, chunk_end)
    rank_samples = max(0, rank_end - rank_start)

    existing = sorted(
        int(f.split("_")[0])
        for f in os.listdir(logger.get_dir())
        if f.endswith("_ckpt.jpg")
    )
    valid_ids = [idx for idx in existing if rank_start <= idx < rank_end]
    resume_start = rank_start
    if valid_ids:
        resume_start = max(valid_ids) + 1
    resume_start = min(resume_start, rank_end)
    rank_samples_remaining = max(0, rank_end - resume_start)

    logger.log(
        f"Rank {rank}/{world_size}: "
        f"chunk_range=[{chunk_start}, {chunk_end}), "
        f"rank_range=[{rank_start}, {rank_end}), "
        f"resume_start={resume_start}, "
        f"generating {rank_samples_remaining} samples."
    )
    if valid_ids:
        logger.log(
            f"Rank {rank}: Resuming from {resume_start} "
            f"(already found {len(valid_ids)} images in assigned range)"
        )

    # 7. Sampling Loop
    generated_count = 0
    batch_size = int(cfg.inference.batch_size)
        
    while generated_count < rank_samples_remaining:
        
        # Adjust last batch size if needed
        current_bs = min(batch_size, rank_samples_remaining - generated_count)
        
        model_kwargs = {}
        if cfg.model.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(current_bs,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.ddim_sample_loop if cfg.inference.use_ddim else diffusion.p_sample_loop
        )

        extra_kwargs = {}
        sample = sample_fn(
            model,
            (current_bs, 3, cfg.model.image_size, cfg.model.image_size),
            clip_denoised=cfg.inference.clip_denoised,
            model_kwargs=model_kwargs,
            **extra_kwargs,
        )

        # Normalize and Save
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()

        for i, arr in enumerate(sample):
            global_index = resume_start + generated_count + i
            fname = (
                f"{global_index:06d}_"
                f"{cfg.model.image_size}x{cfg.model.image_size}x3_"
                f"{cfg.inference.inference_checkpoint}_ckpt.jpg"
            )
            
            Image.fromarray(arr.cpu().numpy()).save(os.path.join(logger.get_dir(), fname))

        generated_count += current_bs
        
        logger.log(f"Rank {rank}: Generated {generated_count}/{rank_samples_remaining} ")

    # 8. Cleanup
    dist.barrier() 
    logger.log("Sampling complete.")
    

if __name__ == "__main__":
    main()