import os
from typing import Any, Dict, Optional, Tuple

import torch
import torchvision

from torch import Tensor
from tqdm import tqdm

import wandb

from .sde import SDE

from src.train_eval_setups.diff_model_recon.diffmodels.networks import UNetModel

def _ddim(
    s: Tensor,
    xhat: Tensor,
    ts: Tuple[Tensor, Tensor],
    sde: SDE,
    eta: float
    ) -> Tensor:

    current_time, previous_time = ts
    mean_prev_time = sde.marginal_prob_mean(
            t=previous_time)[:, None, None, None]
    mean_curr_time = sde.marginal_prob_mean(
            t=current_time)[:, None, None, None]
    
    sqrt_beta = ((1 - mean_prev_time.pow(2)) / (1 - mean_curr_time.pow(2))).sqrt() * \
                (1 - mean_curr_time.pow(2) / mean_prev_time.pow(2)).sqrt()
    if sqrt_beta.isnan().any():
        sqrt_beta = torch.zeros_like(sqrt_beta, device= s.device)
    scaled_noise = xhat * mean_prev_time
    deterministic_noise = torch.sqrt(1 - mean_prev_time.pow(2) - sqrt_beta.pow(2) * eta**2) *  s
    stochastic_noise = eta * sqrt_beta * torch.randn_like(xhat)

    return scaled_noise + deterministic_noise + stochastic_noise

def _wrap_ddim(
    score: UNetModel,
    x: Tensor,
    t: Tuple[Tensor, Tensor],
    sde: SDE,
    **kwargs
    ) -> Tuple[Tensor, Tensor]:
    
    with torch.no_grad():
        s = score(x, t[0]).detach()
        xhat0 = _atweedy(
            s=s,
            x=x,
            t=t[0],
            sde=sde
            )
        x = _ddim(
            s=s,
            xhat=xhat0,
            ts=t,
            sde=sde,
            eta=kwargs['eta']
            )

    return x.detach(), xhat0.detach()

def _atweedy(s: Tensor, x: Tensor, sde: SDE, t: Tensor) -> Tensor:

    div = sde.marginal_prob_mean(t)[:, None, None, None].pow(-1)
    std_t = sde.marginal_prob_std(t)[:, None, None, None]
    update = x - s * std_t

    return update * div


def _schedule_jump(num_steps: int, travel_length: int = 1, travel_repeat: int = 1):
    jumps = {}
    for j in range(0, num_steps - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = num_steps
    time_steps = []
    while t >= 1:
        t = t - 1
        time_steps.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                time_steps.append(t)
    time_steps.append(-1)

    return time_steps

class DDIM():

    def __init__(
        self,
        score: UNetModel,
        sde: SDE,
        sample_kwargs: Dict,
        device: Optional[Any] = None,
        save_img_each_sample_iteration: Optional[int] = None
        ) -> None:
        
        self.score = score
        self.sde = sde
        self.sample_kwargs = sample_kwargs
        self.device = device
        self.predictor = _wrap_ddim
        self.initialise()

        self.save_img_each_sample_iteration = save_img_each_sample_iteration
    
    def initialise(self, ): 

        assert self.sde.num_steps >= self.sample_kwargs['num_steps']
        skip = self.sde.num_steps // self.sample_kwargs['num_steps']

        ts = _schedule_jump(self.sample_kwargs['num_steps'])
        time_pairs = list(
            (i * skip , j * skip if j > 0 else -1)
            for i, j in zip(ts[:-1], ts[1:])
        )        
        self.time_pairs = time_pairs

    def sample(self) -> Tensor:

        init_x = self.sde.prior_sampling(
            [self.sample_kwargs['batch_size'], *self.sample_kwargs['im_shape']]
        ).to(self.device)

        x = init_x
        i = 0
        pbar = tqdm(self.time_pairs)
        for step in pbar:
            ones_vec = torch.ones(
                self.sample_kwargs['batch_size'],
                device=self.device
            )
            t = (ones_vec * step[0], ones_vec * step[1])

            x, x_mean = self.predictor(
                score=self.score,
                x=x,
                t=t,
                sde=self.sde,
                **self.sample_kwargs,
            )

            if self.save_img_each_sample_iteration is not None: 
                if i % self.save_img_each_sample_iteration == 0 or i == self.sample_kwargs['num_steps'] - 1:
                    from src.train_eval_setups.diff_model_recon.utils.wandb_utils import tensor_to_wandbimage
                    wandb.log(
                        {
                            'reco': tensor_to_wandbimage(x_mean.norm(dim=-3)), 
                            'global_step': i,
                        }
                    )
            i += 1

        return x_mean
