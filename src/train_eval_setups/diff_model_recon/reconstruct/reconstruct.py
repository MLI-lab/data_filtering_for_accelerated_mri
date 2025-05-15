import logging
import torch
import wandb

from ..utils.logger import sample_logger_gen
from functools import partial

from omegaconf import DictConfig
from ..physics_trafos import get_standard_trafo_fwd, get_standard_target_trafo, get_standard_prior_trafo
from ..reconstruct import fit, get_criterion
from ..utils import ScoreWithIdentityGradWrapper

from ..diffmodels.networks import load_score_model
from ..diffmodels import load_sde_model
from ..utils.metrics import PSNR
from ..utils.wandb_utils import tensor_to_wandbimages_dict
import math
import numpy as np
import time

from ..physics_trafos.trafo_utils import get_standard_dataset_transform

from fastmri.data.transforms import complex_center_crop, center_crop

def reconstruction_setup(cfg : DictConfig, device : str):
    """
        Inputs:
            - cfg
            - eval_dataset_config_model
            - eval_dataset_config_paths_base
        Outputs:
            - forward_fn function
            - dataset_trafo
    """
    score, sde = None, None
    if cfg.reconstruct.use_score_regularisation: 
        score = load_score_model(cfg, device)#.to(device)
        sde = load_sde_model(cfg)
        #wandb.run.summary['num_params_score'] = sum(p.numel() for p in score.parameters() if p.requires_grad)
    if cfg.reconstruct.use_score_pass_through and cfg.reconstruct.use_score_regularisation: 
        score = ScoreWithIdentityGradWrapper(score=score)
    dataset_trafo = get_standard_dataset_transform(cfg,
        device=device,
        provide_pseudinverse=True,
        provide_measurement=True
    )
    forward_fn = partial(
        reconstruct,
        # cfg=cfg,
        score=score,
        sde=sde,
        device=device
    )
    
    return forward_fn, dataset_trafo

def reconstruct(sample, cfg : DictConfig, score, sde, device) -> None:

    logging.info(f'Using device {device}')
    dtype = torch.get_default_dtype()

    # load trafos which determine the setup (e.g. 3D MRI vs 2D MRI)
    # trafo: object to measurement data (e.g. complex fourier data or sinogram)
    fwd_trafo = get_standard_trafo_fwd(cfg).to(dtype=dtype, device=device)
    # trafo: object to target data (e.g. complex image to magnitude, if we compare to magnitude images)
    target_trafo = get_standard_target_trafo(cfg)
    # trafo: reconstruction object to prior (complex image to magnitude images, if the diffusion model is trained on magnitude)
    prior_trafo = get_standard_prior_trafo(cfg)
            
    # Todo: this we might want to change later
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed) 

    # this needs to be adapted
    observation, ground_truth, filtbackproj, attrs = sample
    # we assume that the batch size during eval is 1
    # so we first squeeze the batch dimension and later unsquezee it before return
    assert observation.shape[0] == 1 and ground_truth.shape[0] == 1 and filtbackproj.shape[0] == 1, "Batch size during eval should be 1."
    observation = observation.to(dtype=dtype, device=device).squeeze()
    filtbackproj = filtbackproj.to(dtype=dtype, device=device).squeeze()
    ground_truth = ground_truth.to(dtype=dtype, device=device).squeeze()

    if cfg.reconstruct.logging.save_observation:
        torch.save(observation, f'observation.pt')
    if cfg.reconstruct.logging.save_filtbackproj:
        torch.save(filtbackproj, f'filtbackproj.pt')
    if cfg.reconstruct.logging.save_ground_truth:
        torch.save(ground_truth, f'ground_truth.pt')

    rep_shape = filtbackproj.shape

    # calibrate trafo (e.g. sensitivity maps)
    fwd_trafo.calibrate(observation, attrs)    

    if cfg.reconstruct.rescale_observation:
        scaling_factor = math.sqrt(float(np.prod(rep_shape).item())) / observation.detach().cpu().norm() * cfg.reconstruct.constant_scaling_factor
    else:
        scaling_factor = cfg.reconstruct.constant_scaling_factor

    trafo_adjoint = fwd_trafo.trafo_adjoint(observation)
    print('trafo_adjoint', trafo_adjoint.shape)

    adjoint_psnr = PSNR(center_crop(target_trafo(trafo_adjoint), ground_truth.shape[-2:]).squeeze(0), ground_truth)

    wandb.log({
        'adj_psnr' : adjoint_psnr,
        'scaling_factor' : scaling_factor,
        **tensor_to_wandbimages_dict("ground_truth", ground_truth.unsqueeze(0), take_meanslices=True, take_videos=False, show_phase=cfg.reconstruct.logging.show_phase),
        **tensor_to_wandbimages_dict("fbp", filtbackproj.unsqueeze(0), take_meanslices=True, take_videos=False, show_phase=cfg.reconstruct.logging.show_phase),
        }
    )
    trafo_adjoint = None; torch.cuda.empty_cache()

    sample_logger = partial(sample_logger_gen, fwd_trafo=fwd_trafo, target_trafo=target_trafo, observation=observation, ground_truth=ground_truth, scaling_factor=scaling_factor, device=device, **cfg.reconstruct.logging)
    
    if cfg.reconstruct.method == 'variational': ## DURING EVAL
        ####################################
        ### Variational methods
        ####################################
        initialise_with = None
        if cfg.reconstruct.use_filterbackproj_as_init: 
            initialise_with = torch.clone(filtbackproj) * scaling_factor # / filtbackproj.std()
            logging.info("initialise with pseudoinverse")
            print('initialise_with', initialise_with.shape)
        elif cfg.reconstruct.use_l1wavelet_as_init:
            from ..utils.bart_utils import compute_l1_wavelet_solution
            logging.info("calculating l1 wavelet solution as init")
            l1_wavelet_solution = compute_l1_wavelet_solution(observation, attrs["sens_maps"], reg_param=4e-4)
            initialise_with = l1_wavelet_solution * scaling_factor
            logging.info(f"L1w - solution has PSNR of {PSNR(l1_wavelet_solution, ground_truth)}")
            print('initialise_with', initialise_with.shape)
        
        # set filtbackproj to None to save memory
        filtbackproj = None; torch.cuda.empty_cache()

        # create reconstruction tensor which is updated during reconstruction
        representation = torch.nn.Parameter(
                torch.randn(rep_shape, device=device) if initialise_with is None else initialise_with, requires_grad=True
            )

        # get slice method (which takes slices of 3D volumes for regularization)
        from .slice_methods import get_slice_method
        slice_method_prior_reg = get_slice_method(slice_cfg=cfg.reconstruct.slice_method_prior_reg) if cfg.reconstruct.slice_method_prior_reg is not None else None

        # get the criterion (e.g. the variational objective)
        criterion = get_criterion(
            trafo_fwd=fwd_trafo,
            observation=observation * scaling_factor,
            steps_data_con=cfg.reconstruct.variational_optim.gradient_acc_steps_data_con,
            steps_data_reg=cfg.reconstruct.variational_optim.gradient_acc_steps_prior_reg,
            prior_trafo=prior_trafo,
            score=score, 
            sde=sde,
            slice_method_prior_reg=slice_method_prior_reg,
            outer_iterations_max=cfg.reconstruct.variational_optim.iterations,
            diffusion_reg_params=cfg.reconstruct.diffusion_reg_params
        )

        # minimize the variational objective
        final_representation = fit(
            representation=representation, 
            criterion=criterion,
            optim_kwargs=cfg.reconstruct.variational_optim,
            sample_logger=sample_logger,
            save_img_each_sample_iteration=cfg.reconstruct.logging.foreach_num_im_in_sample_log
        )
        # final representation has shape
        #  - (Y, X, 2) for 2D MRI
        #  - (Z, Y, X, 2) for 3D MRI
        recon = final_representation.detach().unsqueeze(0) / scaling_factor

    else:
        raise NotImplementedError(f'Reconstruction method {cfg.reconstruct.method} not implemented.')

    if cfg.reconstruct.logging.save_final_sample:
        torch.save(recon, f"final_rec.pt")

    return target_trafo(recon), recon