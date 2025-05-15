from typing import Optional

import torch

from torch import Tensor

from src.train_eval_setups.diff_model_recon.diffmodels.networks import UNetModel
from src.train_eval_setups.diff_model_recon.diffmodels import DDPM
import math

def noise_loss(
    output: Tensor,
    outer_iteration : int,
    outer_iterations_max : int,
    score: UNetModel,
    sde: DDPM,
    repetition : int = 1,
    reg_strength: float = 1.,
    steps_scaler : float = 0.5,
    time_sampling_method : str = 'random',
    adapt_reg_strength: Optional[bool] = None
    ) -> Tensor:
    
    output = output.repeat(repetition, *[1]*(output.ndim -1))
    if time_sampling_method == 'random':
        t = torch.randint(1, 
            math.floor(steps_scaler * sde.num_steps),
            (output.shape[0],),
            device=output.device
        ) # random time-sampling (allows for batching and single time step reg.)
    elif time_sampling_method == 'linear_descending':
        t = torch.tensor(min(max(
                math.floor(
                    float(steps_scaler) * sde.num_steps * (outer_iterations_max - outer_iteration) / outer_iterations_max  # where 100 is max number of iterations
                ), 0), sde.num_steps - 1), device=output.device).repeat(output.shape[0])
    else:
        raise NotImplementedError(f'time_sampling {time_sampling_method} not implemented')
    z = torch.randn_like(output)
    mean, std = sde.marginal_prob(output, t)
    perturbed_x = mean + z * std[:, None, None, None]
    zhat = score(perturbed_x, t)

    if perturbed_x.size(1) == 1 and zhat.size(1) == 2:
        # this occurs when learn_sigma is enabled for the trained network
        zhat = zhat[:, :1]

    mean = torch.mean((z - zhat).pow(2))

    reg_strength_t = reg_strength
    if adapt_reg_strength: 
        # See Mardani et al. (2023), bar_a is not the bar_a from DDPM's definition here
        bar_a = sde.marginal_prob_mean(t)
        reg_strength_t = std / bar_a * reg_strength

    return mean * reg_strength_t