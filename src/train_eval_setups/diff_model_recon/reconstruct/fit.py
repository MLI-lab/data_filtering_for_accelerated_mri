from typing import Dict, Tuple, Union, Optional, Any

import torch
import numpy as np
import wandb

from torch import Tensor
from tqdm import tqdm

from src.train_eval_setups.diff_model_recon.physics_trafos.trafos_prior_target.BasePriorTrafo import BasePriorTrafo

def fit(
    representation,
    criterion: callable,
    optim_kwargs: Dict = {},
    sample_logger : Optional[Any] = None,
    save_img_each_sample_iteration : int = 1.0
    ) -> Tensor:
        
        optim_kwargs = optim_kwargs or {}
        optim_kwargs.setdefault('lr', 1e-4)
        optim_kwargs.setdefault('iterations', 10000)

        optimizer = torch.optim.Adam(
            [representation], lr=optim_kwargs['lr'],
            betas=(0.9, 0.99), weight_decay=0.0 #, eps=1e-15
            )

        if optim_kwargs['lr_scheduler_name'] is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **optim_kwargs['lr_scheduler_params'])
        else:
             scheduler = None

        # list of steps, e.g. data con might be [0, 1] and data reg might be [2, 3]
        max_steps_data_reg = max(optim_kwargs['gradient_acc_steps_prior_reg']) if len(optim_kwargs['gradient_acc_steps_prior_reg']) > 0 else 0
        max_iteration = max(max(optim_kwargs['gradient_acc_steps_data_con']), max_steps_data_reg)

        with tqdm(range(optim_kwargs['iterations']), desc='coord-net') as pbar:

            for i in pbar:

                optimizer.zero_grad()

                datafit_it = torch.zeros(1, device=representation.device)
                regfit_it = torch.zeros(1, device=representation.device)
                for j in range(0, max_iteration+1):
                    loss, datafit, regfit = criterion(representation, i, j)
                    datafit_it += datafit
                    regfit_it += regfit
                    loss.backward()
                loss_it = datafit_it + regfit_it

                if optim_kwargs['clip_grad_max_norm'] is not None:
                    torch.nn.utils.clip_grad_norm_(representation, max_norm=optim_kwargs['clip_grad_max_norm'])

                scheduler.step(loss_it.item())

                if save_img_each_sample_iteration is not None and sample_logger is not None:
                    if (i+1) % save_img_each_sample_iteration == 0 or i == optim_kwargs['iterations'] - 1:
                        with torch.no_grad():
                            sample_logger(representation=representation,#.detach(), # min_loss_sample=min_loss_state['output'].detach(),
                                step=i+1, pbar=pbar, 
                                        log_dict={"loss" : loss_it.item(), "datafit" : datafit_it, "regfit" : regfit_it, "lr" : optimizer.param_groups[0]['lr']})

                optimizer.step()
        
        return representation