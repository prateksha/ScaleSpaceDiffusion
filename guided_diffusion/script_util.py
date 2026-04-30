import inspect

from . import gaussian_diffusion as gd
from . import scale_space_diffusion as ssd
from .respace import SpacedDiffusion, space_timesteps, SpacedScaleSpaceDiffusion
from .unet import UNetModel, FlexiUNet
from . import logger
from . global_config import DEFAULT_CHANNEL_MULTS
from omegaconf import OmegaConf
NUM_CLASSES = 1000
VALID_EXPERIMENT_TYPES = {"ddpm", "ssd"}

def create_model_and_diffusion(cfg, resolutions_array, unet_res_layer_map):
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    diffusion_config = OmegaConf.to_container(cfg.diffusion, resolve=True)
    ssd_config = OmegaConf.to_container(cfg.ssd, resolve=True)
    inference_cfg = cfg.get("inference")
    inferencing_flag = bool(inference_cfg and inference_cfg.get("inferencing_flag", False))
    runtime_cfg = inference_cfg if inferencing_flag else cfg.training

    model_config.update({
        "ssd_config": ssd_config,
        "unet_res_layer_map": unet_res_layer_map,
        "learn_sigma": cfg.diffusion.learn_sigma, 
    })
    model = create_model(**model_config)

    diffusion_config.update({
        "experiment_root_dir": cfg.experiment.root_dir,
        "experiment_name": cfg.experiment.name,
        "experiment_type": cfg.experiment.experiment_type,
        "ssd_config": ssd_config,
        "resolutions_array": resolutions_array,
        "image_size": cfg.model.image_size,
        "dump_visualization_interval": cfg.training.get("dump_visualization_interval", 0),
               "inference_checkpoint": (
            runtime_cfg.get("inference_checkpoint", None) if inferencing_flag else None
        ),
    })
    diffusion = create_gaussian_diffusion(**diffusion_config)

    return model, diffusion

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    ssd_config=None,
    unet_res_layer_map=None,
):
    if channel_mult == "":
        if image_size in DEFAULT_CHANNEL_MULTS:
            channel_mult = tuple(float(ch_mult) for ch_mult in DEFAULT_CHANNEL_MULTS[image_size].split(","))
        else:
            raise ValueError(f"Unsupported image size: {image_size}")
    else:
        channel_mult = tuple(float(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    shared_model_kwargs = dict(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )

    use_flexiunet = False #for standard
    if ssd_config:
        mode = ssd_config.get("multires_training_mode", "")
        if mode == "flexi_unet":
            use_flexiunet = True
    
    if use_flexiunet:
        logger.log("Creating FlexiUNet model")
        return FlexiUNet(
            **shared_model_kwargs,
            unet_res_layer_map=unet_res_layer_map,
        )
    else:
        logger.log("Creating UNetModel")
        return UNetModel(**shared_model_kwargs)

def create_gaussian_diffusion(
    *,
    diffusion_steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    experiment_root_dir="",
    experiment_name="",
    experiment_type="ddpm",
    dump_visualization_interval=0,
    inference_checkpoint=None,
    ssd_config=None,
    resolutions_array=None,
    mse_loss_weight_type='constant',
    image_size=None,
):
    experiment_type = str(experiment_type).lower()
    if experiment_type not in VALID_EXPERIMENT_TYPES:
        raise ValueError(
            f"Invalid experiment_type '{experiment_type}'. "
            f"Expected one of: {sorted(VALID_EXPERIMENT_TYPES)}"
        )

    if experiment_type == "ssd":
        betas = ssd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        loss_type = ssd.LossType.MSE
  
        if not timestep_respacing:
            timestep_respacing = [diffusion_steps]

        assert predict_xstart == True
        model_mean_type = ssd.ModelMeanType.START_X_RS
        model_var_type = ssd.ModelVarType.FIXED_SMALL # does not matter


        return SpacedScaleSpaceDiffusion(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            experiment_root_dir=experiment_root_dir,
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            dump_visualization_interval=dump_visualization_interval,
            inference_checkpoint=inference_checkpoint,
            ssd_config=ssd_config,
            resolutions_array=resolutions_array,
            mse_loss_weight_type=mse_loss_weight_type,
        )
    else:
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        if not timestep_respacing:
            timestep_respacing = [diffusion_steps]
    
        return SpacedDiffusion(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            experiment_root_dir=experiment_root_dir,
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            dump_visualization_interval=dump_visualization_interval,
            inference_checkpoint=inference_checkpoint,
            ssd_config=ssd_config,
            resolutions_array=resolutions_array,
            mse_loss_weight_type=mse_loss_weight_type,
        )
