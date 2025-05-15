import torch
from tqdm.autonotebook import tqdm
from fastmri.evaluate import ssim, psnr
from runstats import Statistics
import pandas as pd
from torch.nn.functional import interpolate
from fastmri.data.transforms import center_crop
from pathlib import Path
import os
import lpips
from DISTS_pytorch import DISTS
import json
import uuid
from datetime import date
import yaml
import numpy as np
import wandb
import random
import math

from omegaconf import DictConfig, OmegaConf
from src.train_eval_setups.diff_model_recon.utils.wandb_utils import wandb_kwargs_via_cfg
from src.train_eval_setups.diff_model_recon.utils.device_utils import get_free_cuda_devices
from src.train_eval_setups.diff_model_recon.reconstruct.reconstruct import reconstruction_setup


import matplotlib.pyplot as plt
import fastmri

import contextlib
import io

from fastmri.evaluate import ssim, psnr
from scipy import interpolate


def get_normalization_params(output, target, mask=None):
    if mask is not None: # not usable for multibatch eval
        mt = target[mask].mean()[None,None,None,None]
        st = target[mask].std()[None,None,None,None]
        mo = output[mask].mean()[None,None,None,None]
        so = output[mask].std()[None,None,None,None]
    else:
        mt = target.mean(axis=(-2, -1), keepdims=True)
        st = target.std(axis=(-2, -1), keepdims=True)
        mo = output.mean(axis=(-2, -1), keepdims=True)
        so = output.std(axis=(-2, -1), keepdims=True)
        
    scale = st/so
    shift = mt - mo * st/so
    return scale, shift

def normalize(x, scale, shift, mask=None):
    if mask is not None:
        x[mask] = x[mask] * scale + shift
        return x
    else:
        return x * scale + shift

def suppress_prints():
    # Create a dummy stream to redirect stdout
    return contextlib.redirect_stdout(io.StringIO())
def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)

def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)

def val_reg_tuning(sample, forward_fn, cfg, device):
    if cfg.hp_tuning.name == "cv_bisection":
        return val_reg_tuning_cv_bisection(sample, forward_fn, cfg, device, **cfg.hp_tuning.params)
    elif cfg.hp_tuning.name == "overall":
        return val_reg_tuning_overall(sample, forward_fn, cfg, device, **cfg.hp_tuning.params)
    elif cfg.hp_tuning.name == "oa_bisection":
        return val_reg_oa_bisection(sample, forward_fn, cfg, device, **cfg.hp_tuning.params)
    elif cfg.hp_tuning.name == "fixed_reg_value":
        # here we just return the fixed reg_strength
        return cfg.reconstruct.diffusion_reg_params.reg_strength
    elif cfg.hp_tuning.name == "cv_sim_annealing":
        return val_reg_tuning_cv_simulated_annealing(sample, forward_fn, cfg, device, **cfg.hp_tuning.params)
    elif cfg.hp_tuning.name == "tuning_gt":
        return gt_reg_tuning(sample, cfg, forward_fn, device, **cfg.hp_tuning.params)
    elif cfg.hp_tuning.name == "tuning_cv":
        return val_reg_tuning_cv(sample, forward_fn, cfg, device, **cfg.hp_tuning.params)
    else:
        raise ValueError(f"Unknown tuning method: {cfg.hp_tuning.name}")

def gt_reg_tuning(sample, cfg, forward_fn, device, 
    # reg_params = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    reg_params = None
    ):
    print("GT spline tuning")
    reg_params = [0.00055, 0.001, 0.0055, 0.01, 0.055, 0.1]
    cfg['reconstruct']['variational_optim']['iterations'] = 100
    _, target, _, attrs = sample
    maxval = attrs["max"].item()
    target = target.to(device)
    output_mask = attrs['output_mask'].to(device)
    output_mask = center_crop(output_mask, target.shape[-2:])
    output_mask = output_mask[0]
    target = target * output_mask
    output_mask = output_mask.unsqueeze(0)
    ssim_record = []
    for reg in reg_params:
        cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = reg
        with suppress_prints():
            output = forward_fn(sample, cfg)
        mag_output = output[0]
        output = center_crop(mag_output, target.shape[-2:])
        output = output.unsqueeze(0)
        output = output * output_mask

        scale, shift = get_normalization_params(output, target.unsqueeze(0), output_mask)
        output = normalize(output, scale, shift, output_mask)
        ssim_record.append(ssim(output[0].cpu().numpy(),target.cpu().numpy(), maxval).item())

    x = np.log10(np.array(reg_params))
    y = np.array(ssim_record)
    f = interpolate.interp1d(x, y, kind='cubic')
    x_test = np.linspace(x.min(), x.max(), 10000)
    reg = 10**x_test[np.argmax(f(x_test))].item()
    
    return reg

def val_reg_tuning_cv(sample, forward_fn, cfg, device, 
    validation_line_ration = 0,
    # reg_params = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]
    reg_params = [0.05,0.1,0.15,0.2,0.25,0.3,0.35],
    # reg_params = [0.02,0.04,0.06,0.08,0.1,0.12]
    validation_times = 1):
    
    masked_kspace, target, pseudorecon, attrs = sample
    masked_image = complex_mul(fastmri.ifft2c(masked_kspace), complex_conj(torch.view_as_real(attrs['smaps'])).to(device)).sum(dim=1, keepdim=False)
    masked_kspace_sc = fastmri.fft2c(masked_image).squeeze()
    
    mask1d = attrs['mask'].squeeze()
    sample_index = torch.where(mask1d==1)[0]
    validation_lines_list = [np.random.choice(sample_index.cpu(),int(np.ceil(len(sample_index)*validation_line_ration))) for i in range(5)]
    loss_record = []
    for reg in reg_params:
        loss = 0
        for i in range(validation_times):
            # validation_lines = np.random.choice(sample_index.cpu(),int(np.ceil(len(sample_index)*validation_line_ration)))
            validation_lines = validation_lines_list[i]
            # Set the validation kspace lines to 0.
            masked_masked_kspace = masked_kspace.clone()
            masked_masked_kspace[...,validation_lines,:] = 0
            sample_val = [masked_masked_kspace, target, pseudorecon, attrs]
            
            cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = reg
            with suppress_prints():
                output = forward_fn(sample_val,cfg)
            complex_output = output[1].squeeze()
            complex_output_kspace = fastmri.fft2c(complex_output)
            # Calculate the Loss:
            # loss += (torch.sum((complex_output_kspace[:,validation_lines,:]-masked_kspace_sc[:,validation_lines,:])**2)/(torch.sum(masked_kspace_sc[:,validation_lines,:]**2)+1e-8)).item()
            loss += (torch.sum((complex_output_kspace[:,sample_index,:]-masked_kspace_sc[:,sample_index,:])**2)/(torch.sum(masked_kspace_sc[:,sample_index,:]**2)+1e-8)).item()
        loss_record.append(loss/validation_times)

    return reg_params[loss_record.index(min(loss_record))]

def val_reg_oa_bisection(sample, forward_fn, cfg, device,
    max_steps = 9 ,
    reg_params_bound = [0.01,0.49],
    validation_times = 1,
    variational_optim_iterations_hp = 150):
    
    cfg['reconstruct']['variational_optim']['iterations'] = variational_optim_iterations_hp
    
    masked_kspace, target, pseudorecon, attrs = sample
    masked_image = complex_mul(fastmri.ifft2c(masked_kspace), complex_conj(torch.view_as_real(attrs['smaps'])).to(device)).sum(dim=1, keepdim=False)
    if complex_output == 3:
            complex_output_kspace = fastmri.fft2c(complex_output)
    else:
        from src.train_eval_setups.diff_model_recon.physics_trafos.trafo_datasets.fftn3d import fft3c
        masked_kspace_sc = (fft3c(masked_image)*attrs['mask']).squeeze()
    
    loss_bound = []
    for reg in reg_params_bound:
        cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = reg
        with suppress_prints():
            output = forward_fn(sample,cfg)
        complex_output = output[1].squeeze()
        if complex_output == 3:
            complex_output_kspace = fastmri.fft2c(complex_output)
        else:
            complex_output_kspace = fft3c(complex_output)
        complex_output_kspace = (complex_output_kspace * attrs['mask']).squeeze()
        loss = ((complex_output_kspace - masked_kspace_sc).square().sum() / (masked_kspace_sc.square().sum() + 1e-8)).item()

        loss_bound.append(loss)
        
    for i in range(max_steps-2):
        if loss_bound[0]>=loss_bound[1]:
            reg_params_bound[0] = (reg_params_bound[0]+reg_params_bound[1])/2
            reg = reg_params_bound[0]
        else:
            reg_params_bound[1] = (reg_params_bound[0]+reg_params_bound[1])/2
            reg = reg_params_bound[1]
        cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = reg
        with suppress_prints():
            output = forward_fn(sample,cfg)
        complex_output = output[1].squeeze()
        # Calculate the Loss:
        if complex_output == 3:
            complex_output_kspace = fastmri.fft2c(complex_output)
        else:
            complex_output_kspace = fft3c(complex_output)
        complex_output_kspace = (complex_output_kspace * attrs['mask']).squeeze()
        loss = ((complex_output_kspace - masked_kspace_sc).square().sum() / (masked_kspace_sc.square().sum() + 1e-8)).item()

        if loss_bound[0]>=loss_bound[1]:
            loss_bound[0] = loss
        else:
            loss_bound[1] = loss
    
    return reg_params_bound[loss_bound.index(min(loss_bound))]

def gt_reg_tuning_cv_bisection(sample, forward_fn, cfg, device, 
    max_steps = 0,
    validation_line_ration = 0,
    reg_params_bound = 0,
    validation_times = 0,
    variational_optim_iterations_hp = 0
    ):
    print("BISECTION TUNING")
    # cfg['reconstruct']['variational_optim']['iterations'] = variational_optim_iterations_hp
    cfg['reconstruct']['variational_optim']['iterations'] = 100
    
    _, target, _, attrs = sample
    maxval = attrs["max"].item()


    loss_bound = []

    max_steps = 7
    reg_params_bound = [-3.301, -0.301]
    for reg in reg_params_bound:
        reg = 10**reg
        cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = reg
        with suppress_prints():
            output = forward_fn(sample, cfg)
        mag_output = output[0]
        output = center_crop(mag_output, target.shape[-2:])
        # Calculate the Loss:    
        loss = -ssim(output.cpu().numpy(),target.cpu().numpy(), maxval).item()
        loss_bound.append(loss)
    
    for _ in range(max_steps-2):
        if loss_bound[0]>=loss_bound[1]:
            reg_params_bound[0] = (reg_params_bound[0]+reg_params_bound[1])/2
            reg = reg_params_bound[0]
        else:
            reg_params_bound[1] = (reg_params_bound[0]+reg_params_bound[1])/2
            reg = reg_params_bound[1]
        reg = 10**reg
        cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = reg
        with suppress_prints():
            output = forward_fn(sample, cfg)
        mag_output = output[0]
        output = center_crop(mag_output, target.shape[-2:])
        # Calculate the Loss:    
        loss = -ssim(output.cpu().numpy(),target.cpu().numpy(), maxval).item()
        if loss_bound[0]>=loss_bound[1]:
            loss_bound[0] = loss
        else:
            loss_bound[1] = loss

    reg = 10**reg_params_bound[loss_bound.index(min(loss_bound))]

    # with open('/kang/data_filtering_mri/biscetion_log_scale_result_new.txt', 'a') as f:
    #     f.write(f'{reg}, ')

    # import csv
    # cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = reg
    # with suppress_prints():
    #     output = forward_fn(sample, cfg)
    # mag_output = output[0]
    # output = center_crop(mag_output, target.shape[-2:])
    # ssim_value = ssim(output.cpu().numpy(),target.cpu().numpy(),maxval).round(3).item()
    # psnr_value = psnr(output.cpu().numpy(),target.cpu().numpy(),maxval).round(2)
    # result = [reg, ssim_value, psnr_value]
    # with open('/kang/data_filtering_mri/gt_bisection_result_new.csv', 'a') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=",")
    #     writer.writerow(result)

    return reg

def val_reg_tuning_cv_bisection(sample, forward_fn, cfg, device,
    max_steps = 9 ,
    validation_line_ration = 0,
    reg_params_bound = [0.01,0.49],
    validation_times = 1,
    variational_optim_iterations_hp = 150
    ):
    
    # cfg['reconstruct']['variational_optim']['iterations'] = variational_optim_iterations_hp
    cfg['reconstruct']['variational_optim']['iterations'] = 50
    
    masked_kspace, target, pseudorecon, attrs = sample
    masked_image = complex_mul(fastmri.ifft2c(masked_kspace), complex_conj(torch.view_as_real(attrs['smaps'])).to(device)).sum(dim=1, keepdim=False)
    masked_kspace_sc = fastmri.fft2c(masked_image).squeeze()
    
    # mask = attrs['mask']
    # squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
    # cent = squeezed_mask.shape[1] // 2
    # # running argmin returns the first non-zero
    # left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
    # right = torch.argmin(squeezed_mask[:, cent:], dim=1)
    # print(squeezed_mask[0])
    # print('HALLO', cent,left, right)
    
    mask1d = attrs['mask'].squeeze().clone()
    
    # print(mask1d.shape)
    # mask1d[:cent-left] = 0
    # mask1d[cent+right:] = 0
    
    sample_index = torch.where(mask1d==1)[0]
    validation_lines_list = [np.random.choice(sample_index.cpu(),int(np.ceil(len(sample_index)*validation_line_ration))) for i in range(5)]
    loss_bound = []
    _, coils, _, _, _ = masked_kspace.shape
    print(f'Number of coils {coils}')
    
    if coils <= 4: 
        reg_params_bound = [-2.3, -0.52] # log [0.005, 0.3]
    elif coils <= 8: 
        reg_params_bound = [0.005, 0.05]
    elif coils <= 16: 
        reg_params_bound = [0.005, 0.03]
    else: 
        reg_params_bound = [0.002, 0.01]
        
    # reg_params_bound = [0.001, 0.5]
    for reg in reg_params_bound:
        if coils <=4:
            reg=10**reg
        loss = 0
        for i in range(validation_times):
            # validation_lines = np.random.choice(sample_index.cpu(),int(np.ceil(len(sample_index)*validation_line_ration)))
            validation_lines = validation_lines_list[i]
            # Set the validation kspace lines to 0.
            masked_masked_kspace = masked_kspace.clone()
            masked_masked_kspace[...,validation_lines,:] = 0
            sample_val = [masked_masked_kspace, target, pseudorecon, attrs]
            cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = reg
            with suppress_prints():
                output = forward_fn(sample_val,cfg)
            complex_output = output[1].squeeze()
            complex_output_kspace = fastmri.fft2c(complex_output)
            # Calculate the Loss:
            # loss += (torch.sum((complex_output_kspace[:,validation_lines,:]-masked_kspace_sc[:,validation_lines,:])**2)/(torch.sum(masked_kspace_sc[:,validation_lines,:]**2)+1e-8)).item()
            loss += (torch.sum((complex_output_kspace[:,sample_index,:]-masked_kspace_sc[:,sample_index,:])**2)/(torch.sum(masked_kspace_sc[:,sample_index,:]**2)+1e-8)).item()
        # loss_record.append(loss/validation_times)
        loss_bound.append(loss/validation_times)
    
    for i in range(max_steps-2):
        if loss_bound[0]>=loss_bound[1]:
            reg_params_bound[0] = (reg_params_bound[0]+reg_params_bound[1])/2
            reg = reg_params_bound[0]
        else:
            reg_params_bound[1] = (reg_params_bound[0]+reg_params_bound[1])/2
            reg = reg_params_bound[1]
        if coils <=4:
            reg=10**reg
        loss = 0
        for i in range(validation_times):
            # validation_lines = np.random.choice(sample_index.cpu(),int(np.ceil(len(sample_index)*validation_line_ration)))
            validation_lines = validation_lines_list[i]
            # Set the validation kspace lines to 0.
            masked_masked_kspace = masked_kspace.clone()
            masked_masked_kspace[...,validation_lines,:] = 0
            sample_val = [masked_masked_kspace, target, pseudorecon, attrs]
            
            cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = reg
            with suppress_prints():
                output = forward_fn(sample_val,cfg)
            complex_output = output[1].squeeze()
            complex_output_kspace = fastmri.fft2c(complex_output)
            # Calculate the Loss:
            # loss += (torch.sum((complex_output_kspace[:,validation_lines,:]-masked_kspace_sc[:,validation_lines,:])**2)/(torch.sum(masked_kspace_sc[:,validation_lines,:]**2)+1e-8)).item()
            loss += (torch.sum((complex_output_kspace[:,sample_index,:]-masked_kspace_sc[:,sample_index,:])**2)/(torch.sum(masked_kspace_sc[:,sample_index,:]**2)+1e-8)).item()
        # loss_record.append(loss/validation_times)
        # loss_bound.append(loss/validation_times)
        # if (loss/validation_times)>max(loss_bound):
        #     break
        if loss_bound[0]>=loss_bound[1]:
            loss_bound[0] = loss/validation_times
        else:
            loss_bound[1] = loss/validation_times
    if coils <= 4:
        return 10**reg_params_bound[loss_bound.index(min(loss_bound))]
    else:
        return reg_params_bound[loss_bound.index(min(loss_bound))]

def val_reg_tuning_overall(sample, forward_fn, cfg, device,
    reg_params = [0.05,0.1,0.15,0.2,0.25,0.3,0.35]):
    
    masked_kspace, _, _, attrs = sample
    masked_image = complex_mul(fastmri.ifft2c(masked_kspace), complex_conj(torch.view_as_real(attrs['smaps'])).to(device)).sum(dim=1, keepdim=False)
    masked_kspace_sc = fastmri.fft2c(masked_image).squeeze()
    
    mask1d = attrs['mask'].squeeze()
    sample_index = torch.where(mask1d==1)[0]

    loss_record = []
    for reg in reg_params:
        cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = reg
        with suppress_prints():
            output = forward_fn(sample,cfg)
        complex_output = output[1].squeeze()
        complex_output_kspace = fastmri.fft2c(complex_output)
        # Calculate the Loss:
        loss_record.append((torch.sum((complex_output_kspace[:,sample_index,:]-masked_kspace_sc[:,sample_index,:])**2)/(torch.sum(masked_kspace_sc[:,sample_index,:]**2))+1e-8).item())
    
    return reg_params[loss_record.index(min(loss_record))]


# Other tuning methods:
def val_reg_tuning_cv_simulated_annealing(sample, forward_fn, cfg, device, max_steps,
    validation_line_ration = 0,
    # Define the initial search range for the regularization parameter
    reg_params_bound = [0.001, 1],
    # Set the initial validation times and the number of iterations for the inner optimization
    validation_times = 1,
    variational_optim_iterations_hp = 150
    ):
    cfg['reconstruct']['variational_optim']['iterations'] = variational_optim_iterations_hp
    
    masked_kspace, target, pseudorecon, attrs = sample
    masked_image = complex_mul(fastmri.ifft2c(masked_kspace), complex_conj(torch.view_as_real(attrs['smaps'])).to(device)).sum(dim=1, keepdim=False)
    masked_kspace_sc = fastmri.fft2c(masked_image).squeeze()
    
    mask1d = attrs['mask'].squeeze()
    sample_index = torch.where(mask1d == 1)[0]
    validation_lines_list = [np.random.choice(sample_index.cpu(), int(np.ceil(len(sample_index) * validation_line_ration))) for i in range(5)]
    
    # Initial settings for Simulated Annealing
    current_reg = np.random.uniform(reg_params_bound[0], reg_params_bound[1])  # Start with a random reg parameter within bounds
    best_reg = current_reg
    cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = current_reg
    
    # Calculate the initial loss
    def calculate_loss(reg):
        loss = 0
        for i in range(validation_times):
            validation_lines = validation_lines_list[i]
            masked_masked_kspace = masked_kspace.clone()
            masked_masked_kspace[..., validation_lines, :] = 0
            sample_val = [masked_masked_kspace, target, pseudorecon, attrs]
            
            cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = reg
            with suppress_prints():
                output = forward_fn(sample_val, cfg)
            complex_output = output[1].squeeze()
            complex_output_kspace = fastmri.fft2c(complex_output)
            loss += (torch.sum((complex_output_kspace[:, sample_index, :] - masked_kspace_sc[:, sample_index, :]) ** 2) / (torch.sum(masked_kspace_sc[:, sample_index, :] ** 2) + 1e-8)).item()
        
        return loss / validation_times
    
    # Evaluate the loss for the current regularization parameter
    current_loss = calculate_loss(current_reg)
    best_loss = current_loss
    
    # Initial temperature and cooling rate
    temperature = 1.0  # Starting temperature
    cooling_rate = 0.95  # Cooling factor, which will reduce the temperature in each iteration
    
    for step in range(max_steps):
        # Generate a new candidate reg parameter by adding a small perturbation
        candidate_reg = current_reg + random.uniform(-0.05, 0.05)  # Adjust step size as needed
        # Ensure the candidate_reg is within bounds
        candidate_reg = max(min(candidate_reg, reg_params_bound[1]), reg_params_bound[0])
        
        # Calculate the loss for the candidate solution
        candidate_loss = calculate_loss(candidate_reg)
        
        # Acceptance criterion: always accept if the candidate loss is better
        if candidate_loss < current_loss:
            current_reg = candidate_reg
            current_loss = candidate_loss
            
            # Update the best solution if this candidate is the best seen so far
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_reg = candidate_reg
        else:
            # If the candidate is worse, accept it with a certain probability
            acceptance_probability = math.exp(-(candidate_loss - current_loss) / temperature)
            if random.uniform(0, 1) < acceptance_probability:
                current_reg = candidate_reg
                current_loss = candidate_loss
        
        # Reduce the temperature according to the cooling rate
        temperature *= cooling_rate
        
        # Optionally, print the progress
        print(f"Step {step+1}/{max_steps}, Current Reg: {current_reg}, Loss: {current_loss}, Temperature: {temperature}")

    return best_reg
