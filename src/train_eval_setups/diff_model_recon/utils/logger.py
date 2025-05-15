# %%
from abc import ABC, abstractmethod
from torch import Tensor

import os
from typing import Any, Dict, Optional, Tuple, List

import torch
import torchvision

from torch import Tensor
from tqdm import tqdm
import math

from fastmri.data.transforms import center_crop

import wandb
import numpy as np

from src.train_eval_setups.diff_model_recon.diffmodels.sde import SDE


from src.train_eval_setups.diff_model_recon.utils.metrics import PSNR, PSNR_2D, SSIM, VIFP
from src.train_eval_setups.diff_model_recon.utils.utils import midslice2selct, normalize_clamp, align_normalization
from src.train_eval_setups.diff_model_recon.utils.wandb_utils import tensor_to_wandbimages_dict

def sample_logger_pass(sample : Tensor, min_loss_sample : Tensor, step : int, pbar, log_dict : Dict):
    pass

def sample_logger_simple(sample : Tensor, min_loss_sample : Tensor, step : int, pbar, log_dict : Dict):
    wandb.log(
        {
            'rec_mean': sample.detach().cpu().numpy().mean(),
            'rec_std': sample.detach().cpu().numpy().std(),
            'global_step': step,
            **tensor_to_wandbimages_dict("reco", sample, show_phase=False),
            **log_dict
        }
    )
    #pbar.set_description(f'rec_psnr={sample_psnr:.1f}, rec_ssim={sample_ssim:.2f}', refresh=False)

def sample_logger_gen(representation : Tensor, step : int, pbar, log_dict : Dict, fwd_trafo, target_trafo, observation, ground_truth, scaling_factor, show_phase, device, sample_statistics_period : 5, sample_statistic_log_image_period : 50, mean_slice_period : int = 5, video_period : int = 300, sample_statistics_take_medslices : bool = False, log_3d_metrics_from_slices : bool = True, log_data_is_complex : bool = True, log_psnr_magnitude : bool = True, log_3d_slice_metrics_use_vol_max : bool = False, take_abs_normalize : bool = False, **kwargs):

    with torch.no_grad():
        sample = representation # .forward_splitted(mesh, device, sample_gen_split)

        tf_sample = target_trafo(sample) / scaling_factor
        gt = ground_truth.cpu().to(device) # problems on servers..

        # 3D PSNR
        sample_psnr = PSNR(center_crop(tf_sample, gt.shape[-2:]), gt)
        #sample_ssim_mean, sample_ssim_std = SSIM(target_trafo(sample).norm(dim=-1) / scaling_factor, ground_truth.cpu().norm(dim=-1))

    #sample_ssim = SSIM(target_trafo(sample).cpu() / scaling_factor, ground_truth.cpu())
    #sample_vifp = VIFP(target_trafo(sample).cpu() / scaling_factor, ground_truth.cpu())

    extra_dict = {}
    #if log_3d_metrics_from_slices:
        #with torch.no_grad():

            #tf_sample_ssim = tf_sample.norm(dim=-1) if log_data_is_complex else tf_sample
            #gt_ssim = gt.norm(dim=-1) if log_data_is_complex else gt

            #ssim_dim1_mean, ssim_dim1_std = SSIM(tf_sample_ssim, gt_ssim, axis=0)
            #ssim_dim2_mean, ssim_dim2_std = SSIM(tf_sample_ssim, gt_ssim, axis=1)
            #ssim_dim3_mean, ssim_dim3_std = SSIM(tf_sample_ssim, gt_ssim, axis=2)

            #tf_sample_psnr = tf_sample.norm(dim=-1) if log_data_is_complex and log_psnr_magnitude else tf_sample 
            #gt_psnr = gt.norm(dim=-1) if log_data_is_complex and log_psnr_magnitude else gt

            #psnr_dim1_mean, psnr_dim1_std = PSNR_2D(tf_sample_psnr, gt_psnr, axis=0, use_vol_max=log_3d_slice_metrics_use_vol_max)
            #psnr_dim2_mean, psnr_dim2_std = PSNR_2D(tf_sample_psnr, gt_psnr, axis=1, use_vol_max=log_3d_slice_metrics_use_vol_max)
            #psnr_dim3_mean, psnr_dim3_std = PSNR_2D(tf_sample_psnr, gt_psnr, axis=2, use_vol_max=log_3d_slice_metrics_use_vol_max)

            #extra_dict = {
                #'ssim_dim1_mean': ssim_dim1_mean,
                #'ssim_dim1_std': ssim_dim1_std,
                #'ssim_dim2_mean': ssim_dim2_mean,
                #'ssim_dim2_std': ssim_dim2_std,
                #'ssim_dim3_mean': ssim_dim3_mean,
                #'ssim_dim3_std': ssim_dim3_std,
                #'psnr_dim1_mean': psnr_dim1_mean,
                #'psnr_dim1_std': psnr_dim1_std,
                #'psnr_dim2_mean': psnr_dim2_mean,
                #'psnr_dim2_std': psnr_dim2_std,
                #'psnr_dim3_mean': psnr_dim3_mean,
                #'psnr_dim3_std': psnr_dim3_std
            #}

    if step % sample_statistic_log_image_period == 0:
        extra_dict = {
            **(tensor_to_wandbimages_dict("reco", sample, take_meanslices=step % mean_slice_period == 0 and step > 0, take_videos=step % video_period == 0 and step > 0, show_phase=show_phase)),
        }

    wandb.log(
        {
            #'min_loss_rec_psnr': min_loss_sample_psnr,
            'rec_psnr': sample_psnr,
            #'min_loss_rec_ssim': min_loss_sample_ssim,
            #'rec_ssim': sample_ssim_mean,
            #'min_loss_rec_vfip': min_loss_sample_vifp,
            #'rec_vfip': sample_vifp,
            'rec_mean': sample.detach().cpu().numpy().mean(),
            'rec_std': sample.detach().cpu().numpy().std(),
            'global_step': step,
            **log_dict,
            **extra_dict
        }
    )
    #pbar.set_description(f'rec_psnr={sample_psnr:.1f}, rec_ssim={sample_ssim:.2f}', refresh=False)
    pbar.set_description(f'rec_psnr={sample_psnr:.1f}', refresh=False)

    # if sample_statistics_take_medslices:
    #     #
    #     with torch.no_grad():
    #         sample_mean_slice = representation.forward(mesh.add_index_select(axis=0, indices=torch.Tensor([mesh.matrix_size[0] // 2]).int()))
    #         sample_mean_slice_psnr = PSNR_2D(target_trafo(sample_mean_slice) / scaling_factor, ground_truth[ground_truth.shape[0]//2,...][None].cpu().to(device), take_abs_normalize=take_abs_normalize)[0] # take the 'mean'

    #         #sample_mean_slice_ssim = SSIM(target_trafo(sample_mean_slice).norm(dim=-1) / scaling_factor, ground_truth[ground_truth.shape[0]//2,...].cpu().norm(dim=-1))

    #         #sample_mean_slice_vifp = VIFP(target_trafo(sample_mean_slice).norm(dim=-1) / scaling_factor, ground_truth[ground_truth.shape[0]//2,...].cpu().norm(dim=-1))

    #     wandb.log(
    #         {
    #             'global_step': step,
    #             'rec_medslice_psnr': sample_mean_slice_psnr,
    #             #'rec_medslice_ssim': sample_mean_slice_ssim,
    #             #'rec_medslice_vfip': sample_mean_slice_vifp,
    #             'rec_medslice_mean': sample_mean_slice.detach().cpu().numpy().mean(),
    #             'red_medslice_std': sample_mean_slice.detach().cpu().numpy().std(),
    #             'global_step': step,
    #             **(tensor_to_wandbimages_dict("reco_medslice",sample_mean_slice.unsqueeze(0), show_phase=show_phase)),
    #             **log_dict
    #         }
    #     )
    #     pbar.set_description(f'rec_medslice_psnr={sample_mean_slice_psnr:.1f}', refresh=False)