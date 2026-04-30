import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import re

from guided_diffusion import logger
def torch_resize_batch(img_batch, size):
    img_batch = nn.functional.interpolate(img_batch, size=size, mode='bilinear', align_corners=False, antialias = True)
    return img_batch

def get_resolution_discrete(scale_max, t=0.0, num_steps=1000, r_conversion_mode="equal", model_channel_mult=(1,2,4,8), multires_training_mode=None):
    if not (0 <= t < num_steps):
        raise ValueError(f"t must be in [0, {num_steps-1}], got {t}")

    num_levels = len(model_channel_mult)
    resolutions = [scale_max // (2 ** i) for i in range(num_levels)]

    n = len(resolutions)
    tau = t / (num_steps - 1)

    if r_conversion_mode == "equal":
         idx = min(int(tau * n), n - 1)
    else:
        raise ValueError(f"Unknown r_conversion_mode: {r_conversion_mode}")

    return resolutions[idx]

def find_closest_number_divisible(divisible, num):
    rem = num % divisible
    if rem<=(divisible/2):
        return num-rem
    return num+(divisible-rem)

def get_value_for_resolution(new_res, sizes, values):
    for i in range(len(sizes) - 1):
        if sizes[i] >= new_res > sizes[i + 1]:
            return values[i], sizes[i]

    # Handle edge cases (beyond largest/smallest)
    if new_res > sizes[0]:
        return values[0], sizes[0]
    elif new_res <= sizes[-1]:
        return values[-1], sizes[-1]
    else:
        return None 




import numpy as np
from scipy.optimize import root_scalar

# ---------------------------------------------------------------------
# Helper functions (opposite of the smaller codebase, reversing the naming, this is correct now)
# ---------------------------------------------------------------------
def sharper_ft(t, gamma=2): 
    x = np.sign(t) * np.abs(t)**gamma + 0.5
    return (3 * x**2 - 2 * x**3 - 0.5)

def normalized_sharper_ft(t, gamma=2):
    return sharper_ft(t, gamma=gamma) / sharper_ft(-0.5, gamma=gamma)

def tanh_like(t, gamma=2):
    return 0.5 * normalized_sharper_ft(t - 0.5, gamma=gamma) + 0.5

def invert_normalized_sharper_ft(y, gamma=2):
    def func_to_solve(t):
        return 0.5 * normalized_sharper_ft(t, gamma) - y
    sol = root_scalar(func_to_solve, bracket=[-0.5, 0.5], method='brentq')
    return sol.root if sol.converged else np.nan

invert_tanh_vec = np.vectorize(invert_normalized_sharper_ft)

def normalized_invert_sharper_ft(t, gamma=2):
    return invert_tanh_vec(t, gamma=gamma) / invert_tanh_vec(-0.5, gamma=gamma)

def sigmoid_like(t, gamma=2):
    return 0.5 * normalized_invert_sharper_ft(t - 0.5, gamma=gamma) + 0.5


def get_resolution_scale_space(resolutions, t, num_steps, resolution_schedule="equal"):
    if not (0 <= t < num_steps):
        raise ValueError(f"t must be in [0, {num_steps-1}], got {t}")
    n = len(resolutions)

    if n < 2:
        return resolutions[0]

    tau = t / (num_steps - 1)
    if resolution_schedule == "equal":
        idx = n - 1 - min(int(tau * n), n - 1) 
    elif "ConvexDecay" in resolution_schedule: 
        gamma = float(resolution_schedule.replace("ConvexDecay_", ""))
        x = 1 - (1 - tau) ** gamma
        idx = n - 1 - min(int(x * n), n - 1)
    elif "SigmoidLikeDecay" in resolution_schedule:
        gamma = float(resolution_schedule.replace("SigmoidLikeDecay_", ""))
        x = 1 - sigmoid_like(1 - tau, gamma)  # reversed so high→low
        idx = min(int(x * n), n - 1)
    elif "TanhLikeDecay" in resolution_schedule:
        gamma = float(resolution_schedule.replace("TanhLikeDecay_", ""))
        x = 1 - tanh_like(1 - tau, gamma)
        idx = min(int(x * n), n - 1)
    else:
        raise ValueError(f"Unknown resolution_schedule: {resolution_schedule}")
    return resolutions[idx]

def get_resolutions_array(ssd_config=None,):
    if ssd_config is None:
        logger.info(f"SSD Configuration: {ssd_config}")  
        return None
    logger.info("SSD Configuration:")
    logger.info(
        "\n"
        f"  • ssd_config_flag         : {ssd_config['ssd_config_flag']}\n"
        f"  • resolution_schedule     : {ssd_config['resolution_schedule']}\n"
        f"  • scale_max               : {ssd_config['scale_max']}\n"
        f"  • model_channel_mult      : {ssd_config['model_channel_mult']}\n"
        f"  • t_res_sampling_mode     : {ssd_config['t_res_sampling_mode']}\n"
        f"  • multires_training_mode  : {ssd_config['multires_training_mode']}\n"
        f"  • num_levels              : {ssd_config['num_levels']}\n"
        f"  • ds_factor               : {ssd_config['ds_factor']}\n"
    )
    scale_max = ssd_config["scale_max"]
    model_channel_mult = ssd_config["model_channel_mult"] #(1,2,4,8), 

    if type(model_channel_mult) == str:
        model_channel_mult = tuple(float(ch_mult) for ch_mult in model_channel_mult.split(","))
    multires_training_mode = ssd_config["multires_training_mode"] #None, 
    num_levels = ssd_config["num_levels"] if ssd_config["num_levels"] is not None else 4 #2
    ds_factor = ssd_config["ds_factor"] if ssd_config["ds_factor"] is not None else 2 #2
    resolution_schedule = ssd_config["resolution_schedule"]

    resolutions = [] 
    unet_res_layer_map = {}

    if multires_training_mode=="flexi_unet":
        if num_levels == None and ds_factor == None:
            num_levels = len(model_channel_mult)
            resolutions = [scale_max // (2 ** i) for i in range(num_levels)]
            for res in resolutions:
                unet_res_layer_map[res] = res
        else:

            current_res = scale_max
            channel_mult_length = len(model_channel_mult)

            default_resolutions = [scale_max/pow(2,ds_idx) for (ds_idx, channel_dim) in enumerate(model_channel_mult)]
            divisibles = [pow(2, curr_channel_len-1) for curr_channel_len in range(channel_mult_length, 0, -1)]

            level_idx = 0
            while level_idx<num_levels and current_res>=1 and 1 not in resolutions:
                curr_divisible, curr_high_res = get_value_for_resolution(current_res, default_resolutions, divisibles)
                level_resolution = int(find_closest_number_divisible(curr_divisible, round(current_res)))

                resolutions.append(level_resolution)
                unet_res_layer_map[level_resolution] = curr_high_res

                current_res = round(current_res / ds_factor) # 32 becomes 32.000001 etc
                level_idx +=1

            if num_levels is not None and level_idx < num_levels:
                raise ValueError(
                    f"num_levels={num_levels} is too high. "
                    f"Only {level_idx} levels are possible with scale_max={scale_max}, "
                    f"ds_factor={ds_factor}, and channel_mult_length={channel_mult_length}."
                )

    resolutions.reverse()
    logger.log(f"Resolution list: {resolutions},")
    return resolutions, unet_res_layer_map
