from typing import Tuple, Union, Iterator, List
from itertools import repeat

import torch
import numpy as np

from pathlib import Path
from torch import Tensor
from PIL import Image
import logging

import xml.etree.ElementTree as etree
from copy import deepcopy
import math
from pathlib import Path
from torch.utils.data import ConcatDataset
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import torch
from fastmri import complex_abs

from fastmri.data.transforms import MaskFunc, to_tensor, complex_center_crop, center_crop, normalize_instance, normalize
from .transforms import apply_mask

from .fftn3d import fft3c, ifft3c
import fastmri

from sigpy.mri.samp import poisson, radial, spiral

import torchvision.transforms.functional as TF

class FastMRI3DDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_enabled : bool,
        mask_type : str,
        mask_accelerations : Tuple[int],
        mask_center_fractions : Tuple[float],
        mask_use_seed: bool = True,
        provide_pseudinverse : bool = False,
        provide_measurement : bool = True,
        use_real_synth_data : bool = False,
        return_magnitude_image : bool = False,
        scale_target_by_kspacenorm : bool = False,
        target_scaling_factor : float = 1.0,
        target_interpolate_by_factor : float = 1.0,
        target_interpolation_method : str = "nearest",
        normalize_target : bool = False,
        target_type : str = "rss",
        multicoil_reduction_op : bool = "sum",
        device : str = "cpu",
    ):
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.which_challenge = which_challenge
        self.mask_use_seed = mask_use_seed
        self.provide_pseudinverse = provide_pseudinverse
        self.provide_measurement = provide_measurement
        self.use_real_synth_data = use_real_synth_data
        self.return_magnitude_image = return_magnitude_image
        self.normalize_target = normalize_target
        self.target_type = target_type

        self.scale_target_by_kspacenorm = scale_target_by_kspacenorm
        self.target_scaling_factor = target_scaling_factor
        self.target_interpolate_by_factor = target_interpolate_by_factor
        self.target_interpolation_method = target_interpolation_method

        self.wrapped_2d = False
    
        self.device = device
    
        self.multicoil_reduction_op = multicoil_reduction_op


        self.mask_accelerations = mask_accelerations
        self.mask_use_seed = mask_use_seed 

        self.mask_type = mask_type
        self.mask_enabled = mask_enabled
        if mask_enabled:
            if mask_type == 'Poisson2D':
                pass
            #else:
                #mask_class = fastmri.data.subsample.RandomMaskFunc if mask_type == 'random' else fastmri.data.subsample.EquispacedMaskFractionFunc
                #self.mask_func = mask_class(center_fractions=mask_center_fractions, accelerations=[mask_accelerations], seed=mask_seed)
                #self.seed = mask_seed
            elif self.mask_type == 'random':
                mask_class = fastmri.data.subsample.RandomMaskFunc 
                self.mask_func = mask_class(center_fractions=mask_center_fractions, accelerations=[mask_accelerations])
            elif self.mask_type == 'equispaced':
                mask_class = fastmri.data.subsample.EquispacedMaskFractionFunc
                self.mask_func = mask_class(center_fractions=mask_center_fractions, accelerations=[mask_accelerations])
        else:
            self.mask_func = None

    def __call__(
        self,
        kspace: np.ndarray,
        mask_: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:

        crop_size = (320, 320)


        kspace_torch = to_tensor(kspace) if not torch.is_tensor(kspace) else kspace

        if self.device is not None:
            kspace_torch = kspace_torch.to(self.device)

        # first the target
        if target is not None:

            if self.target_type == "mvue" or self.target_type == "rss":
                target_torch = to_tensor(target) if not torch.is_tensor(target) else target

                if self.device is not None:
                    target_torch = target_torch.to(self.device)

            elif self.target_type == "recalc_target":
                if not self.wrapped_2d:
                    target_torch = ifft3c(kspace_torch)
                else:
                    target_torch = fastmri.ifft2c(kspace_torch)

                if self.which_challenge == "multicoil":
                    # this should usually be done in the dataloader
                    if self.multicoil_reduction_op == "sum":
                        target_torch = target_torch.sum(dim=0) # ??
                    elif self.multicoil_reduction_op == "mean":
                        target_torch = target_torch.mean(dim=0)
                    elif self.multicoil_reduction_op == "norm":
                        target_torch = target_torch.norm(dim=0)
                    elif self.multicoil_reduction_op == "norm_sum_sensmaps":
                        S = torch.from_numpy(attrs["sens_maps"]).movedim(-1,0).to(self.device)
                        S[S == 0] = S[S != 0].abs().min() + 0j
                        S_normalization = torch.abs(S).square().sum(dim=0).sqrt()
                        target_torch = torch.view_as_real(torch.sum(torch.view_as_complex(target_torch) * torch.conj(S), dim=0) / S_normalization)
                    else:
                        raise NotImplementedError(f"Reduction operation {self.multicoil_reduction_op} not supported")

            if self.scale_target_by_kspacenorm:
                target_torch =  target_torch * math.sqrt(float(np.prod(target_torch.shape).item())) / attrs["kspace_vol_norm"] #kspace_torch.norm()
            
            if self.target_scaling_factor != 1.0:
                target_torch = target_torch * self.target_scaling_factor

            if self.target_interpolate_by_factor != 1.0:
                target_torch = torch.nn.functional.interpolate(target_torch.movedim(-1, 0).unsqueeze(0), scale_factor=self.target_interpolate_by_factor, mode=self.target_interpolation_method).squeeze(0).movedim(0,-1).contiguous()

            if self.normalize_target:
                target_torch, mean, std = normalize_instance(target_torch, eps=1e-11)
                target_torch = target_torch.clamp(-6, 6)
        else:
            target_torch = torch.Tensor([0])

        # then the kspace
        if self.provide_pseudinverse or self.provide_measurement:

            if self.use_real_synth_data:
                assert self.wrapped_2d == False, "Wrapped 2D not supported for real synth data"
                #target_torch = to_tensor(target) if not torch.is_tensor(target) else target
                if self.target_type == "rss":
                    kspace_torch = fft3c(torch.stack([target_torch, torch.zeros_like(target_torch)], dim=-1))
                elif self.target_type == "fullysampled_rec":    
                    kspace_torch = fft3c(target_torch) # use sensitivity matrix?
                else:
                    raise NotImplementedError(f"Target type {self.target_type} not supported")

            if self.target_interpolate_by_factor != 1.0:

                assert not self.wrapped_2d, "Wrapped 2D not supported for interpolating target"
                logging.info(f"Target interpolated by factor {self.target_interpolate_by_factor} and requires measurement or pseudoinverse -> interpolate sensitivities and kspace")

                S_reform = torch.view_as_real(torch.from_numpy(attrs["sens_maps"])).permute(-2, -1, 0, 1, 2).to(self.device)
                S_interpolated = torch.nn.functional.interpolate(
                    S_reform,
                    scale_factor=self.target_interpolate_by_factor,
                    mode=self.target_interpolation_method).moveaxis(1, -1).contiguous() # (Coils, Z', Y', X', 2)
                S_interpolated = torch.view_as_complex(S_interpolated)
                S_new = S_interpolated.moveaxis(0,-1).contiguous()
                attrs["sens_maps"] = S_new.cpu().numpy()
                logging.info(f"sense_map: {self.device}")

                x_sens = torch.view_as_real(
                    torch.view_as_complex(target_torch.unsqueeze(0)) * S_interpolated
                )
                kspace_torch = fft3c(x_sens)

            if self.mask_enabled:

                seed = None if not self.mask_use_seed else tuple(map(ord, fname))
                if self.mask_type == 'Poisson2D':

                    print(f"mask accelerations: {self.mask_accelerations}")
                    ms1 = int(kspace.shape[-3] * self.target_interpolate_by_factor)
                    ms2 = int(kspace.shape[-2] * self.target_interpolate_by_factor)
                    self.mask = torch.from_numpy(poisson([ms1, ms2], self.mask_accelerations, seed=self.mask_use_seed))
                    self.mask_func = None

                    masked_kspace = torch.view_as_real(torch.view_as_complex(kspace_torch) * self.mask.to(kspace_torch.get_device()))
                else:
                    masked_kspace, mask, _ = apply_mask(kspace_torch, self.mask_func, seed=seed)
                    self.mask = mask

                attrs["mask"] = self.mask # TODO: A better approach is to return the mask like in the varnet data transform from fastMRI codebase where the mask is still used during reconstruction
            else:
                masked_kspace = kspace_torch

        # and finally the pseudorec
        if self.provide_pseudinverse:
            if not self.wrapped_2d:
                image = ifft3c(masked_kspace)
            else:
                image = fastmri.ifft2c(masked_kspace)

            ## check for FLAIR 203
            if image.shape[-2] < crop_size[1]:
                crop_size = (image.shape[-2], image.shape[-2])

            if self.return_magnitude_image:
                # absolute value
                image = fastmri.complex_abs(image)

                ## apply Root-Sum-of-Squares if multicoil data
                if self.which_challenge == "multicoil":
                    image = fastmri.rss(image)
                image = image.unsqueeze(-1)

            elif self.which_challenge == "multicoil" and not self.use_real_synth_data:
                # (15, 640, 320, 2) -> (640, 320, 2)
                if self.multicoil_reduction_op == "sum":
                    image = image.sum(dim=0)
                elif self.multicoil_reduction_op == "mean":
                    image = image.mean(dim=0)
                elif self.multicoil_reduction_op == "norm":
                    image = image.norm(dim=0)
                elif self.multicoil_reduction_op == "norm_sum_sensmaps":
                    S = torch.from_numpy(attrs["smaps"]).movedim(1,0).to(self.device) # by convention smaps is (Z, Coils, Y, X)
                    image = torch.view_as_real(torch.sum(torch.view_as_complex(image) * torch.conj(S), dim=0))
                else:
                    raise NotImplementedError(f"Reduction operation {self.multicoil_reduction_op} not supported")

        # normalize target
        if self.device != "cpu":
            torch.cuda.empty_cache()

        if self.provide_pseudinverse:
            if self.provide_measurement:
                if self.target_type == "mvue":
                    target_torch = complex_abs(target_torch) #.permute(1,2,0))
                    if 'recon_size' in attrs:
                        target_torch = center_crop(target_torch, attrs['recon_size'][:-1])
                return masked_kspace, target_torch, image, attrs
            else:
                return target_torch, image
        else:
            if self.provide_measurement:
                return masked_kspace, target_torch
            else:
                return target_torch