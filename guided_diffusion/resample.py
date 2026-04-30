from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist
from .ssd_utils import *
from collections import defaultdict
import json
import pprint

def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def prepare_sampling(self, ssd_config, resolutions_array):
        """Precompute reusable sampling structures."""
        w = self.weights()
        p = w / np.sum(w)
        self.resolutions_array = resolutions_array
        self.ssd_config = ssd_config
        ssd_t_res_sampling_mode = ssd_config.get("t_res_sampling_mode") if ssd_config else None

        if ssd_t_res_sampling_mode == "one_res_many_t_except_one_t_for_res_change":
            self.t_res_dict = self._build_t_res_dict(ssd_config, resolutions_array, len(p))
            self.res_t_dict = None
        elif ssd_t_res_sampling_mode is not None:
            raise ValueError(
                f"Unknown configuration combination!\n"
                f"  > ssd.t_res_sampling_mode: '{ssd_t_res_sampling_mode}'\n"
                f"No sampling logic matches this pairing."
            )
        else:
            self.t_res_dict = None
            self.res_t_dict = None

    def _should_log_resolution_staging(self, current_itr, progress_breakdown):
        if current_itr is None:
            return False
        if current_itr < 10:
            return True
        if current_itr % 100 == 0:
            return True
        return current_itr % progress_breakdown == 0

    def _build_t_res_dict(self, ssd_config, resolutions_array, len_p,):
        """Build mapping from t → [possible resolutions]."""
        ssd_t_res_sampling_mode = ssd_config.get("t_res_sampling_mode") if ssd_config else None

        if ssd_t_res_sampling_mode == "one_res_many_t_except_one_t_for_res_change":
            num_steps = len_p
            t_res_dict = {}
            for t_i in range(0, num_steps):
                res_t = get_resolution_scale_space(
                    resolutions_array,
                    t=t_i,
                    num_steps=num_steps,
                    resolution_schedule=ssd_config["resolution_schedule"],
                )
                t_res_dict[t_i] = res_t

        return t_res_dict

    def _broadcast_unet_path(self, device, path_key):
        if not dist.is_initialized():
            return tuple(int(v) for v in path_key)

        if dist.get_rank() == 0:
            path_tensor = th.tensor(path_key, device=device, dtype=th.long)
        else:
            path_tensor = th.empty(2, device=device, dtype=th.long)
        dist.broadcast(path_tensor, src=0)
        return tuple(int(v) for v in path_tensor.tolist())

    def sample(self, batch_size, device, ssd_config, resolutions_array, current_itr=None):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        resolutions_selected = None
        
        # --- Case 1: No SSD config (standard random t-sampling)
        if ssd_config is None: 
            indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
            indices = th.from_numpy(indices_np).long().to(device)

            weights_np = 1 / (len(p) * p[indices_np])
            weights = th.from_numpy(weights_np).float().to(device)
        
        # -- Case: If res change, use the same t, else different t's for same res
        elif ssd_config["t_res_sampling_mode"] == "one_res_many_t_except_one_t_for_res_change":
            index_np = np.random.choice(len(p), size=1, p=p)

            chosen_t = index_np.item()
            chosen_s = max(chosen_t - 1, 0)
            rt = self.t_res_dict[chosen_t]
            rs = self.t_res_dict[chosen_s]

            if rt != rs:
                # If resolution change, use the same t for all samples in the batch
                indices = th.full((batch_size,), index_np.item(), dtype=th.long).to(device)  
                weights_np = 1 / (len(p) * p[index_np])
                weights = th.full((batch_size,), weights_np.item(), dtype=th.float).to(device)
            else:
                # If resolution same, use different t's for different samples in the batch
                all_t = np.array(list(self.t_res_dict.keys()))
                all_rt = np.array([self.t_res_dict[t_val] for t_val in all_t])
                all_s = np.maximum(all_t - 1, 0)
                all_rs = np.array([self.t_res_dict[s_val] for s_val in all_s])
                
                ## find indices where the resolution is not chaninging, ie. rs == rt
                valid_t = all_t[(all_rt == rt) & (all_rs == rt)]
                valid_p = p[valid_t]
                valid_p = valid_p / np.sum(valid_p)

                indices_np = np.random.choice(valid_t, size=(batch_size,), p=valid_p)
                indices = th.from_numpy(indices_np).long().to(device)

                weights_np = 1 / (len(p) * p[indices_np])
                weights = th.from_numpy(weights_np).float().to(device)
                
        else:
            raise ValueError(f"Unknown t_res_sampling_mode: {ssd_config['t_res_sampling_mode']}")

        logger.debug(
            f"[DEBUG] Sampled batch info (printitng 10):\n"
            f"  indices = {pprint.pformat(indices.tolist()[:10], indent=4, width=120)}\n"
            f"  weights = {pprint.pformat(weights.tolist()[:10], indent=4, width=120)}\n"
            f"  resolutions_selected = {pprint.pformat(resolutions_selected.tolist()[:10] if resolutions_selected is not None else None, indent=4, width=120)}"
        )

        return indices, weights, resolutions_selected



class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
