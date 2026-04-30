"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

from functools import partial
import re
import enum
import math
import pprint

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from .ssd_utils import *
from .ssd_math_util import BilinearOp, M_0_t, sample_from_simplified_sigma_batched

import os
import torchvision
from . import logger
import torch.distributed as dist

TRAINING_VISUALIZATION_NUM = int(os.environ.get("TRAINING_VISUALIZATION_NUM", 4))  # default 4
INFERENCE_VISUALIZATION_NUM = int(os.environ.get("INFERENCE_VISUALIZATION_NUM", 4))  # default 4

class ExperimentType(enum.Enum):
    SCALE_SPACE = "scale_space"


def _validate_experiment_type(experiment_type):
    et = str(experiment_type).lower()
    if et != "ssd":
        raise ValueError(
            f"ScaleSpaceDiffusion is only for experiment_type='ssd', got '{experiment_type}'"
        )
    return et

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    START_X_RS = enum.auto()  # the model predicts x_0



class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class ScaleSpaceDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        experiment_root_dir="",
        experiment_name="",
        experiment_type="ssd",
        dump_visualization_interval=0,
        inference_checkpoint=None,
        ssd_config=None,
        resolutions_array=None,

        ## SSD
        # resize_operator="BilinearOp",
        ## x0 specific
        mse_loss_weight_type="constant", # this is the epsilon default, use min_snr for x0
    ):
        logger.log(
            f"[LOGGING] scale_space_diffusion_math/ScaleSpaceDiffusion_init:\n"
            f"  experiment_root_dir = {experiment_root_dir}\n"
            f"  experiment_name = {experiment_name}\n"
            f"  experiment_type = {experiment_type}\n"
            f"  ssd_config = {pprint.pformat(ssd_config, indent=4, width=100)}\n"
            f"  resolutions_array = {pprint.pformat(resolutions_array, indent=4, width=100)}"
        )

        self.experiment_root_dir = experiment_root_dir
        self.experiment_name = experiment_name
        _validate_experiment_type(experiment_type)
        self.experiment_type = ExperimentType.SCALE_SPACE
        self.ssd_config = ssd_config
        self.dump_visualization_interval = dump_visualization_interval
        self.training_viz_path = f"{self.experiment_root_dir}/vizualizations/training/"
        self.inference_viz_path = f"{self.experiment_root_dir}/vizualizations/inference/"
        self.resolutions_array = resolutions_array
        if inference_checkpoint is not None:
            self.inference_checkpoint = str(inference_checkpoint) #f"{inference_checkpoint:06d}"
            logger.log(f"In Scale Space Diffusion: setting path for inference: {self.inference_viz_path}/{self.inference_checkpoint}")

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )


        self.resize_operator = self.ssd_config.get("resize_operator", "BilinearOp")
        if self.resize_operator not in ["BilinearOp"]:
            raise ValueError(f"Invalid resize_operator: {self.resize_operator}")

        self.ops = {}
        for t_idx in range(0, self.num_timesteps): # this codebase uses t=0 to T-1
            s_idx=max(t_idx-1, 0)

            rt = self.get_resolution(t_idx)
            rs = self.get_resolution(s_idx)

            H_in = W_in = rs
            H_out = W_out = rt

            if self.resize_operator == "BilinearOp":
                op = BilinearOp(H_in, W_in, H_out, W_out, align_corners=False, antialias=True)

            # ops[t]: M_t: rs -> rt
            self.ops[t_idx] = op
        
        self.cummulative_ops = {}
        for t_idx in range(0, self.num_timesteps): # this codebase uses t=0 to T-1
            # M_{0:t} = M_t M_{t-1} ... M_0
            self.cummulative_ops[t_idx] = partial(M_0_t, self.ops, t_idx)

        self.mse_loss_weight_type = mse_loss_weight_type

    
    def get_resolution(self, t_idx):
        """
        Returns r given t
        It precomputes the resolution base on the self.ssd_config["resolution_schedule"]
        and stores it in a dictionary for fast retrieval.
        :param t: the timestep
        :return: resolution at timestep t
        """
        if not hasattr(self, "t2res_map"):
            self.t2res_map = {}
            for t_i in range(0, self.num_timesteps):
                res_t = get_resolution_scale_space(
                    self.resolutions_array,
                    t=t_i,
                    num_steps=self.num_timesteps,
                    resolution_schedule=self.ssd_config["resolution_schedule"]
                )
                self.t2res_map[t_i] = res_t
        return self.t2res_map[t_idx]

    
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        if dist.get_rank() == 0 and self.dump_visualization_interval>0:
            self._log_single_image_inference(x_start[:INFERENCE_VISUALIZATION_NUM].detach().cpu(), self.inference_checkpoint, t[0].cpu(), "x0" )
            self._log_single_image_inference(x_t[:INFERENCE_VISUALIZATION_NUM].detach().cpu(), self.inference_checkpoint, t[0].cpu(), "xt" )
            self._log_single_image_inference(( _extract_into_tensor(self.posterior_mean_coef1, t, x_start.shape) * x_start)[:INFERENCE_VISUALIZATION_NUM].detach().cpu(), self.inference_checkpoint, t[0].cpu(), "x0_contribution" )
            self._log_single_image_inference(( _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t)[:INFERENCE_VISUALIZATION_NUM].detach().cpu(), self.inference_checkpoint, t[0].cpu(), "xt_contribution" )
        dist.barrier()

        if self.experiment_type == ExperimentType.SCALE_SPACE and \
            self.model_mean_type == ModelMeanType.START_X_RS:


            s = (t-1).clamp(min=0)

            t_idx = t[0].int().item()
            s_idx = s[0].int().item()

            rt = self.get_resolution(t_idx)
            rs = self.get_resolution(s_idx)

            signal_coeffs = self.sqrt_alphas_cumprod
            noise_coeffs = self.sqrt_one_minus_alphas_cumprod


            mu_s = _extract_into_tensor(signal_coeffs, s, x_start.shape) * x_start


            B = x_start.shape[0]
            shape_one = (B, 1, 1, 1)
            c_val = float(signal_coeffs[t_idx] / signal_coeffs[s_idx])
            c = th.as_tensor(c_val, device=x_t.device, dtype=x_t.dtype)

            if rt != rs:
                logger.debug("t_idx, s_idx, rt, rs", t_idx, s_idx, rt, rs)
                logger.debug("M.H_in, M.H_out", self.ops[t_idx].H_in, self.ops[t_idx].H_out)
                logger.debug("mu_s.shape:", mu_s.shape)
                logger.debug("self.ops[t_idx].M(mu_s).shape:", self.ops[t_idx].M(mu_s).shape)
                logger.debug("self.ops[t_idx].MT(x_t - c * self.ops[t_idx].M(mu_s)).shape:", (x_t - c * self.ops[t_idx].M(mu_s)).shape)

            posterior_mean = mu_s + \
                ((_extract_into_tensor(noise_coeffs, s, x_start.shape) / _extract_into_tensor(noise_coeffs, t, x_start.shape))**2) * \
                c * self.ops[t_idx].MT(x_t - c * self.ops[t_idx].M(mu_s))

            if rs == rt:
                sig_s = _extract_into_tensor(noise_coeffs, s, x_start.shape)
                sig_t = _extract_into_tensor(noise_coeffs, t, x_start.shape)
                rho = (sig_s/sig_t)**2
                noise_t_to_s_coeff = sig_s * ((1 - rho * c**2).sqrt())
                posterior_noise = noise_t_to_s_coeff * th.randn_like(x_t, device=x_t.device, dtype=x_t.dtype)
            else:
                shape_hi = (x_t.shape[0], x_t.shape[1], rs, rs) 
                posterior_noise = sample_from_simplified_sigma_batched(
                    M_apply=lambda x: c*self.ops[t_idx].M(x),
                    MT_apply=lambda y: c*self.ops[t_idx].MT(y),
                    sigma_s=noise_coeffs[s_idx],
                    sigma_t=noise_coeffs[t_idx],
                    hi_shape=shape_hi,
                    device=x_t.device,
                    dtype=x_t.dtype,    
                )

            return posterior_mean, posterior_noise
        raise NotImplementedError("Only SCALE_SPACE with START_X_RS is implemented for now.")

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        s = (t-1).clamp(min=0)

        rt = self.get_resolution(t[0].int().item())
        rs = self.get_resolution(s[0].int().item())



        x_t_preprocessed, preprocess_info = self.preprocess_input(x, 
                                        **{
                                            "t": t, 
                                            "s": s,
                                            "rt": rt,
                                            "rs": rs, 
                                        })
        cond_rs = preprocess_info.pop("cond_rs", None) 

        model_output = model(x_t_preprocessed, self._scale_timesteps(t), cond_rs, **model_kwargs)

        model_output, postprocess_info = self.postprocess_output(model_output, 
                                            **{
                                                "rs": rs,
                                                "rt": rt,
                                            })

        if dist.get_rank() == 0 and self.dump_visualization_interval>0:
            self._log_single_image_inference(x[:INFERENCE_VISUALIZATION_NUM].detach().cpu(), self.inference_checkpoint, t[0].cpu(), "model_input" )
            self._log_single_image_inference(model_output[:INFERENCE_VISUALIZATION_NUM].detach().cpu(), self.inference_checkpoint, t[0].cpu(), "model_output" )
        dist.barrier()

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            raise NotImplementedError("Learned variance not implemented for now.")
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X_RS:

            pred_xstart_rs = model_output
            pred_xstart_rs = process_xstart(pred_xstart_rs)
            model_mean, model_noise = self.q_posterior_mean_variance(
                x_start=pred_xstart_rs, x_t=x, t=t
            )

            assert (
                model_mean.shape == model_noise.shape == pred_xstart_rs.shape 
            )
            return {
                "mean": model_mean,
                "pred_xstart": pred_xstart_rs,
                "noise": model_noise,

            }

        raise NotImplementedError("Only START_X_RS is implemented for now.")

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            raise NotImplementedError("cond_fn not implemented for now.")
        
        
        if self.experiment_type == ExperimentType.SCALE_SPACE:
            sample = out["mean"] + nonzero_mask * out["noise"]

        else:
            raise NotImplementedError("Only scale-space diffusion implemented for now.")
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            logger.log(
                "ssd math: p_sample_loop (progressive):\n"
                f"  exp type      : {self.experiment_type}\n"
                f"  exp name      : {self.experiment_name}\n"
                f"  exp root_dir  : {self.experiment_root_dir}"
            )           
            if self.experiment_type == ExperimentType.SCALE_SPACE:
                img = th.randn((shape[0], shape[1], self.ssd_config["scale_min"], self.ssd_config["scale_min"]), device=device)
            elif self.experiment_type == ExperimentType.STANDARD:
                raise NotImplementedError("Standard diffusion not implemented for now.")
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        i=0
        while i<len(indices):

            t = th.tensor([indices[i]] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

                i += 1

    def training_losses(self, model, x_start, t, r, model_kwargs=None, noise=None, step=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if self.experiment_type == ExperimentType.SCALE_SPACE:
            return self._training_losses_scale_space(model, x_start, t, r, model_kwargs, noise, step)
        else:
            raise NotImplementedError(f"Not implemented : {self.experiment_type}")

    def get_mse_loss_weight(self, t):
        ## computing loss weight
        ## base on : https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/main/guided_diffusion/gaussian_diffusion.py#L172
        if self.mse_loss_weight_type == 'constant':
            mse_loss_weight = th.ones_like(t)
        else:
            mse_loss_weight = None
            alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, t.shape)
            sigma = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, t.shape)
            snr = (alpha / sigma) ** 2
            if self.mse_loss_weight_type == 'trunc_snr':
                mse_loss_weight = th.stack([snr, th.ones_like(t)], dim=1).max(dim=1)[0]
            elif self.mse_loss_weight_type == 'snr':
                mse_loss_weight = snr

            elif self.mse_loss_weight_type == 'inv_snr':
                mse_loss_weight = 1. / snr

            elif self.mse_loss_weight_type.startswith("min_snr_"):
                k = float(self.mse_loss_weight_type.split('min_snr_')[-1])
                # min{snr, k}
                mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).min(dim=1)[0]

            elif self.mse_loss_weight_type.startswith("max_snr_"):
                k = float(self.mse_loss_weight_type.split('max_snr_')[-1])
                # min{snr, k}
                mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).max(dim=1)[0]           
            
            if mse_loss_weight is None:
                raise ValueError(f'mse loss weight is not correctly set!')
            # Handle zero-terminal SNR with rescaled betas, where one position could be Infinite:
            mse_loss_weight[snr == 0] = 1.0

        return mse_loss_weight



    def compute_loss(self, pred, target, terms):
        # claculating mse_raw separately otherwise it gets overwritten
        terms["mse_raw"] = mean_flat((target - pred) ** 2)
        # terms[mse] will contain sum of all losses
        terms["mse"] = mean_flat((target - pred) ** 2)
        return terms

    def preprocess_input(self, x, **kwargs):

        cond_rs = None
        if self.ssd_config["multires_training_mode"] == "flexi_unet":
            t = kwargs.pop("t", None)
            s = kwargs.pop("s", None)
            rt = kwargs.pop("rt", None)
            rs = kwargs.pop("rs", None)
            unet_res_layer_map = self.ssd_config["unet_res_layer_map"]
            rs_all = torch.stack([
                torch.tensor(
                    
                    self.get_resolution(t_idx=s_curr.item()),

                    device=s_curr.device,  # optional: keep on same device
                    dtype=torch.float32     # optional: match precision
                )
                for s_curr in s
            ])

            cond_rs = None if rt == rs_all[0] else rs_all

            if cond_rs is not None:
                logger.debug("x_before_resize:", x.shape)
                max_model_res = max(unet_res_layer_map.values())
                if rs == max_model_res:
                    logger.debug(f"special case triggered: rt={rt}, rs={rs}")
                    x = torch_resize_batch(x, rs)
                    cond_rs = None
                logger.debug("x_after_resize:", x.shape)
            
        return x, {"cond_rs":cond_rs}
    
    def postprocess_output(self, model_output, **kwargs):
        if self.ssd_config["multires_training_mode"] == "flexi_unet":
            rt = kwargs.pop("rt", None)
            rs = kwargs.pop("rs", None)
            unet_res_layer_map = self.ssd_config["unet_res_layer_map"]

            if rt != rs:
                expected_output_res = unet_res_layer_map[rs]
                assert expected_output_res == model_output.shape[-1], f"expected output res {expected_output_res}, but got {model_output.shape[-1]}"
                if model_output.shape[-1] != rs:
                    model_output = torch_resize_batch(model_output, rs)
                logger.debug(f"rt={rt} ---> model_output={expected_output_res} ---> rs={rs}")
        return model_output, {}


    def _training_losses_scale_space(self, model, x_start, t, r, model_kwargs=None, noise=None, step=None):
        terms = {}

        s = (t-1).clamp(min=0)
        if (
            self.ssd_config["t_res_sampling_mode"] == "one_t_one_res"
            or self.ssd_config["t_res_sampling_mode"] == "one_res_many_t_except_one_t_for_res_change"
        ):
            rt = self.get_resolution(t_idx=t[0].int().item())
            rs = self.get_resolution(t_idx=s[0].int().item())

        else: 
            raise NotImplementedError(f"Only one_t_one_res is implemented for now. Given: {self.ssd_config['t_res_sampling_mode']}")

        t_idx = t[0].int().item()
        s_idx = s[0].int().item()
        x_start_rt = self.cummulative_ops[t_idx](x_start)
        if s_idx != 0:
            x_start_rs = self.cummulative_ops[s_idx](x_start)
        else:
            x_start_rs = x_start
        
        
        if noise is None:
            noise = th.randn_like(x_start_rt)
        x_t = self.q_sample(x_start_rt, t, noise=noise)

        if self.loss_type == LossType.MSE:
            logger.debug(f"BEFORE MODEL: ", x_t.shape, noise.shape, x_start.shape, x_start_rs.shape, x_start_rt.shape, t.shape, s.shape, rt, rs)
            x_t_preprocessed, preprocess_info = self.preprocess_input(x_t, 
                                                    **{
                                                        "t": t, 
                                                        "s": s,
                                                        "rt": rt,
                                                        "rs": rs,
                                                    })
            cond_rs = preprocess_info.pop("cond_rs", None) 

            model_output = model(x_t_preprocessed, self._scale_timesteps(t), cond_rs, **model_kwargs)
            logger.debug(f"THROUGH MODEL: ", x_t.shape, model_output.shape, noise.shape, type(cond_rs), cond_rs[0] if cond_rs is not None else None)
            model_output, postprocess_info = self.postprocess_output(model_output, 
                                                        **{
                                                            "rs": rs,
                                                            "rt": rt,
                                                        })
            
            if dist.get_rank() == 0 and self.dump_visualization_interval>0 and step % self.dump_visualization_interval == 0:
                self._log_single_image(step, t[0], x_start[:TRAINING_VISUALIZATION_NUM].detach().cpu(), "x0")
                self._log_single_image(step, t[0], x_start_rt[:TRAINING_VISUALIZATION_NUM].detach().cpu(), "x0_rt")
                self._log_single_image(step, t[0], x_t[:TRAINING_VISUALIZATION_NUM].detach().cpu(), "xt")
                self._log_single_image(step, t[0], model_output[:TRAINING_VISUALIZATION_NUM].detach().cpu(), "model_output")
                # self._log_single_image(step, t[0], _pred_x0_rt_from_eps[:TRAINING_VISUALIZATION_NUM].detach().cpu(), "pred_x0_from_eps" )
            dist.barrier()




            target = {
                ModelMeanType.START_X_RS: x_start_rs,
            }[self.model_mean_type]
                
            assert model_output.shape == target.shape == x_start_rs.shape
            mse_loss_weight = self.get_mse_loss_weight(t)

            terms = self.compute_loss(pred=model_output, target=target, terms=terms)

            terms["mse"] = mse_loss_weight * terms["mse"]

            terms["loss"] = terms["mse"]

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _log_single_image(self, step, timestep_val, batch_images, tag):
        image_grid = torchvision.utils.make_grid(batch_images, nrow=2, normalize=True)            
        dir_path = f"{self.training_viz_path}/{step:06d}_{timestep_val:03d}"
        filepath = f"{dir_path}/{tag}_{step}_{timestep_val}.png"

        os.makedirs(dir_path, exist_ok=True)
        torchvision.utils.save_image(image_grid, filepath)

    def _log_single_image_inference(self, images, inference_checkpoint, timestep_val, tag):
        image_grid = torchvision.utils.make_grid(images, nrow=max(int(INFERENCE_VISUALIZATION_NUM/8), 4), normalize=True)
        dir_path = f"{self.inference_viz_path}/{inference_checkpoint}/{timestep_val:03d}/{tag}"
        os.makedirs(dir_path, exist_ok=True)
        filepath = f"{dir_path}/{tag}_{timestep_val}.png"
        torchvision.utils.save_image(image_grid, filepath)
        if INFERENCE_VISUALIZATION_NUM>8:
            for idx, img in enumerate(images):
                filepath = f"{dir_path}/{tag}_{timestep_val}_{idx:03d}.png"
                torchvision.utils.save_image(img, filepath, normalize=True)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
