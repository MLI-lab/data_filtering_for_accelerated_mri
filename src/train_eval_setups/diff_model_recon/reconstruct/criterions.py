from typing import Optional, Dict, Tuple, List, Any
from functools import partial

import torch 

from torch import Tensor

from .noise_loss import noise_loss
from ..diffmodels.networks import UNetModel
from ..diffmodels import DDPM
from ..physics_trafos.trafo_fwd.base_trafo_fwd import BaseRayTrafo
from ..physics_trafos.trafos_prior_target.BasePriorTrafo import BasePriorTrafo
from ..reconstruct.slice_methods import SliceMethod

def _noisereg_criterion(
    representation, 
    outer_iteration : int,
    inner_iteration : int,
    outer_iterations_max : int,
    observation: Tensor,
    steps_data_con : List[int],
    steps_data_reg : List[int],
    fwd_trafo: BaseRayTrafo,
    prior_trafo: BasePriorTrafo,
    score: UNetModel, 
    sde: DDPM,
    slice_method_prior_reg: Optional[SliceMethod],
    diffusion_reg_params: Dict
    ) -> callable:

    criterion = torch.nn.MSELoss()
    # check if in this iteration we consider a data consistency step
    if inner_iteration in steps_data_con:
        datafit = criterion(
            fwd_trafo(representation), 
            observation
        ) / len(steps_data_con)
    else:
        datafit = torch.zeros(1, device=representation.device)

    # check if in this iteration we consider a diffusion reg step.
    if inner_iteration in steps_data_reg:
        # define noise loss
        nl = partial(noise_loss,
            outer_iteration=outer_iteration,
            outer_iterations_max=outer_iterations_max,
            score=score,
            sde=sde, 
            **diffusion_reg_params 
        )

        if slice_method_prior_reg is not None:
            # apply slicing for regularization if enabled
            slices, slice_inds = slice_method_prior_reg(representation, outer_iteration=outer_iteration, inner_iteration=inner_iteration)
        else:
            slices = [representation] 
        regfit = sum([nl(prior_trafo(slice)).mean() for slice in slices]) / len(slices) / len(steps_data_reg)

    else:
        regfit = torch.zeros(1, device=representation.device)

    return datafit + regfit, datafit.item(), regfit.item()

def get_criterion(
    trafo_fwd: BaseRayTrafo,
    observation: Tensor,
    steps_data_con : List[int],
    steps_data_reg : List[int],
    prior_trafo : Optional[BasePriorTrafo] = None,
    score: Optional[UNetModel] = None, 
    sde: Optional[DDPM] = None, 
    slice_method_prior_reg: Optional[SliceMethod] = None,
    diffusion_reg_params: Dict = {},
    outer_iterations_max : Optional[int] = None,
    ) ->  callable:
    
    criterion = partial(
        _noisereg_criterion, 
        outer_iterations_max=outer_iterations_max,
        observation=observation, 
        steps_data_con=steps_data_con,
        steps_data_reg=steps_data_reg,
        fwd_trafo=trafo_fwd,
        prior_trafo=prior_trafo,
        score=score, 
        sde=sde,
        slice_method_prior_reg=slice_method_prior_reg,
        diffusion_reg_params=diffusion_reg_params
    ) 
    return criterion