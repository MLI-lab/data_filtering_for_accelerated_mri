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

def center_pad(a, s):
  """Centers and pads a 3D NumPy array with zeros to a specified size.

  Args:
    a: The input NumPy array of shape (x, y, 2).
    s: The target size for both dimensions.

  Returns:
    The padded array of shape (s, s, 2).
  """

  x, y, _ = a.shape

  # Calculate the padding needed on each side for both dimensions
  pad_x = max((s - x) // 2, 0)
  pad_y = max((s - y) // 2, 0)

  # Create a zero-filled array of the target size
  padded_array = torch.zeros(max(s, x), max(s, y), 2)

  # Calculate the starting indices for placing the original array in the padded array
  start_x = pad_x
  start_y = pad_y

  # Place the original array in the center of the padded array
  padded_array[start_x:start_x+x, start_y:start_y+y] = a

  return padded_array

def random_aug_and_crop(image, size, aug=True):
    if image.shape[0] < size + 2 or image.shape[1] < size + 2:
        image = center_pad(image, size + 2)
        
    if aug:            
        if torch.rand(1) > 0.5: # horizontal flip
            image = torch.flip(image, dims=(-2,))
            
        if torch.rand(1) > 0.5: # vertical flip               
            image = torch.flip(image, dims=(-3,))

        if torch.rand(1) > 0.5: # 90 degree rotation
            image = torch.rot90(image, dims=(-3, -2))

        if torch.rand(1) > 0.5: # swap imag and real
            image = torch.flip(image, dims=(-1, 1))

    # Calculate the maximum possible top-left corner for the crop
    max_top = image.shape[0] - size
    max_left = image.shape[1] - size

    # Randomly select the top-left corner coordinates
    top = np.random.randint(0, max_top+1)
    left = np.random.randint(0, max_left+1)

    # Crop the image
    crop = image[top:top+size, left:left+size, :]


    return crop


class FastMRI2DDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_enabled : bool, mask_type : str, mask_accelerations : Tuple[int],
        mask_center_fractions : Tuple[float],
        mask_use_seed: bool = True,
        provide_pseudinverse : bool = False,
        provide_measurement : bool = True,
        use_real_synth_data : bool = False,
        return_magnitude_image : bool = False,
        return_cropped_pseudoinverse : bool = False,
        scale_target_by_kspacenorm : bool = False,
        scale_target_by_kspacenorm_3d : bool = False,
        target_scaling_factor : float = 1.0,
        target_random_crop_size : Optional[Tuple[int, int]] = None,
        normalize_target : bool = False,
        target_type : str = "rss",
        pseudoinverse_conv_averaging_shape : Optional[Tuple[int, int]] = None,
        multicoil_reduction_op : bool = "sum",
        target_interpolate_by_factor : float = 1.0,
        target_interpolate_factor_is_interval : bool = False,
        target_interpolate_method : str = "nearest",
        device : str = "cpu"
    ):
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.which_challenge = which_challenge
        self.mask_use_seed = mask_use_seed
        self.provide_pseudinverse = provide_pseudinverse
        self.provide_measurement = provide_measurement
        self.use_real_synth_data = use_real_synth_data
        self.return_magnitude_image = return_magnitude_image
        self.return_cropped_pseudoinverse = return_cropped_pseudoinverse
        self.normalize_target = normalize_target
        self.target_type = target_type
        self.target_random_crop_size = target_random_crop_size

        self.scale_target_by_kspacenorm = scale_target_by_kspacenorm
        self.scale_target_by_kspacenorm_3d = scale_target_by_kspacenorm_3d
        self.target_scaling_factor = target_scaling_factor
        self.target_interpolate_by_factor = target_interpolate_by_factor
        self.target_interpolate_factor_is_interval = target_interpolate_factor_is_interval
        self.target_interpolate_method = target_interpolate_method

        self.pseudoinverse_conv_averaging_shape = pseudoinverse_conv_averaging_shape

        self.device = device 
    
        self.multicoil_reduction_op = multicoil_reduction_op

        self.mask_type = mask_type
        self.mask_center_fractions = mask_center_fractions
        self.mask_accelerations = mask_accelerations
        # self.mask_seed = mask_seed
        if mask_enabled:
            if self.mask_type == 'Poisson2D':
                #self.mask = torch.from_numpy(poisson([218, 170], mask_accelerations, seed=mask_seed).astype(np.float32)).unsqueeze(dim=-1)
                pass
                #self.mask = torch.from_numpy(poisson([320, 320], mask_accelerations, seed=mask_seed).astype(np.float32)).unsqueeze(dim=-1)
                self.mask_func = None
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
        if kspace is not None:
            kspace_torch = to_tensor(kspace) if not torch.is_tensor(kspace) else kspace
            if self.device is not None:
                kspace_torch = kspace_torch.to(self.device)
        else:
            kspace_torch = None

        attrs['fname'] = fname
        attrs['slice_num'] = slice_num
        # maxval?



        # sag to cor 
        #kspace_torch = kspace_torch.swapaxes(0, 2)

        #if self.which_challenge == "multicoil":
            # (Z, C, Y, X, 2) -> (C, Z, Y, X, 2)
            #kspace_torch = kspace_torch.moveaxis(0, 1)

        if self.provide_pseudinverse or self.provide_measurement:
            if self.use_real_synth_data:
                # disregard kspace_torch and take fourier transform of target as kspace measurement
                # expand target with additional dimension
                target_torch = to_tensor(target) if not torch.is_tensor(target) else target
                if self.device is not None:
                    target_torch = target_torch.to(self.device)
                kspace_torch = fastmri.fft2c(torch.stack([target_torch, torch.zeros_like(target_torch)], dim=-1))
                # kspace shape (320, 320, 2)

            if self.mask_type == 'Poisson2D':
                print(f"mask shape: {kspace_torch.shape}, acceleration: {self.mask_accelerations}")
                self.mask = torch.from_numpy(poisson(kspace_torch.shape[-3:-1], self.mask_accelerations, seed=self.mask_use_seed).astype(np.float32)).unsqueeze(dim=-1)
                masked_kspace = kspace_torch * self.mask.to(kspace_torch.get_device())
            else:
                seed = None if not self.mask_use_seed else tuple(map(ord, fname))
                masked_kspace, mask, _ = apply_mask(kspace_torch, self.mask_func, seed=seed)
                self.mask = mask

            attrs["mask"] = self.mask # A better approach is to return the mask like in the varnet data transform from fastMRI codebase where the mask is still used during reconstruction

        # inverse Fourier transform to get zero filled solution
        # masked_kspace = masked_kspace#.permute(2, 0, 1) # shange ordering of complex channel

        if self.provide_pseudinverse:
            image = fastmri.ifft2c(masked_kspace)

            ## check for FLAIR 203
            if image.shape[-2] < crop_size[1]:
                crop_size = (image.shape[-2], image.shape[-2])

            if self.return_cropped_pseudoinverse:
                image = complex_center_crop(image, crop_size)

            if self.pseudoinverse_conv_averaging_shape is not None:
                for dim, rep in enumerate(self.pseudoinverse_conv_averaging_shape):
                    image = image.repeat_interleave(repeats=rep, dim=dim)

            if self.return_magnitude_image:
                # absolute value
                image = fastmri.complex_abs(image)

                ## apply Root-Sum-of-Squares if multicoil data
                if self.which_challenge == "multicoil":
                    image = fastmri.rss(image)

                image = image.unsqueeze(-1)

                # normalize input
                #image, mean, std = normalize_instance(image, eps=1e-11)
                #image = image.clamp(-6, 6)
            elif self.which_challenge == "multicoil": ## DURING EVAL
                # (15, 640, 320, 2) -> (640, 320, 2)
                if self.multicoil_reduction_op == "sum":
                    image = image.sum(dim=-4)
                elif self.multicoil_reduction_op == "mean":
                    image = image.mean(dim=-4)
                elif self.multicoil_reduction_op == "norm":
                    image = image.norm(dim=-4)
                elif self.multicoil_reduction_op == "norm_sum_sensmaps":
                    S = torch.from_numpy(attrs["sens_maps"]).to(image.device) # shape is: (Coils, X, Y)
                    S[S == 0] = S[S != 0].abs().min() + 0j # replace zeros with minimum value
                    dim=0
                    S_norm = S.abs().square().sum(dim=dim).sqrt()
                    image = torch.view_as_real(torch.sum(torch.view_as_complex(image) * torch.conj(S), dim=dim) / S_norm)
                else:
                    raise NotImplementedError(f"Reduction operation {self.multicoil_reduction_op} not supported")

        # normalize target
        if target is not None:
            if self.target_type == "rss" or self.target_type == "mvue":
                target_torch = to_tensor(target) if not torch.is_tensor(target) else target
                if self.device is not None:
                    target_torch = target_torch.to(self.device)
                
            elif self.target_type == "fullysampled_rec":
                target_torch = fastmri.ifft2c(kspace_torch)
                #if target_torch.shape[-2] < target[1]:
                    #crop_size = (image.shape[-2], image.shape[-2])
                #target_torch = complex_center_crop(target_torch, crop_size)
                if self.which_challenge == "multicoil":
                # (15, 640, 320, 2) -> (640, 320, 2)
                    #target_torch = target.sum(dim=0)
                    if self.multicoil_reduction_op == "sum":
                        target_torch = target_torch.sum(dim=0) # ??
                    elif self.multicoil_reduction_op == "mean":
                        target_torch = target_torch.mean(dim=0)
                    elif self.multicoil_reduction_op == "norm":
                        target_torch = target_torch.norm(dim=0)
                    elif self.multicoil_reduction_op == "norm_sum_sensmaps":
                        #S = torch.from_numpy(attrs["sens_maps"])
                        #target_torch = torch.sum(target_torch * torch.conj(S), dim=0) / torch.sqrt(torch.sum(torch.square(torch.abs(S)), dim=0))
                        S = torch.from_numpy(attrs["sens_maps"]).to(target_torch.device) # shape is: (Coils, X, Y)
                        if S.abs().sum() == 0.0:
                            target_torch = torch.zeros_like(target_torch).sum(dim=0)
                        else:
                            S[S == 0] = S[S != 0].abs().min() + 0j # replace zeros with minimum value
                            dim=0
                            S_norm = S.abs().square().sum(dim=dim).sqrt()
                            target_torch = torch.view_as_real(torch.sum(torch.view_as_complex(target_torch) * torch.conj(S), dim=dim) / S_norm)
                    else:
                        raise NotImplementedError(f"Reduction operation {self.multicoil_reduction_op} not supported")

            assert not self.scale_target_by_kspacenorm or not self.scale_target_by_kspacenorm_3d, "Only one of the two normalizing forms can be used"

            # This normalizing form might pose problems if we train on targets that are smaller than the k-space
            if self.scale_target_by_kspacenorm: ## THIS
                if kspace_torch is not None:
                    target_torch =  target_torch * math.sqrt(float(np.prod(target_torch.shape).item())) / kspace_torch.norm()
                else:
                    target_torch =  target_torch * math.sqrt(float(np.prod(target_torch.shape).item())) / attrs['kspace_norm']

            if self.scale_target_by_kspacenorm_3d:
                target_torch =  target_torch * math.sqrt(float(np.prod(target_torch.shape).item())) / attrs["kspace_vol_norm"] # for 3D it makes more sense to have the kspace volume norm

            if self.target_scaling_factor != 1.0: ## THIS
                target_torch = target_torch * self.target_scaling_factor
            
            # Is interpolation necesary? I didn't check but potentially some bugs with the added new dimension from mvue, AK: interpolation is not necessary (many other features here are not used), I'll remove them as soon as we fix the model.
            if self.target_interpolate_by_factor is not None: ## THIS
                # test if the interpolate factor is a float
                if isinstance(self.target_interpolate_by_factor, float): ## THIS
                    factor = self.target_interpolate_by_factor
                #elif isinstance(self.target_interpolate_by_factor, List):

                else:
                    if self.target_interpolate_factor_is_interval:
                        # sample random factor between the two interval boundaries (continously)
                        rnd_factor = torch.rand(size=(1,)).item()
                        factor = self.target_interpolate_by_factor[0] + rnd_factor * (self.target_interpolate_by_factor[1] - self.target_interpolate_by_factor[0])
                    else:
                        # sample discretly from the list of factors
                        factor = self.target_interpolate_by_factor[torch.randint(0, len(self.target_interpolate_by_factor), size=(1,)).item()]
                #else:
                    #raise Exception("Interpolate factor must be a float or a list of floats")
    
                target_torch = torch.nn.functional.interpolate(target_torch.movedim(-1, 0).unsqueeze(0), scale_factor=factor, mode=self.target_interpolate_method).squeeze(0).movedim(0,-1)

            if self.target_random_crop_size is not None:
                i = torch.randint(0, target_torch.shape[-3]-self.target_random_crop_size[0] + 1, size=(1,)).item()
                j = torch.randint(0, target_torch.shape[-2]-self.target_random_crop_size[1] + 1, size=(1,)).item()
                target_torch = TF.crop(target_torch.movedim(-1, 0).unsqueeze(0), i, j, self.target_random_crop_size[0], self.target_random_crop_size[1]).squeeze(0).movedim(0, -1)

            if self.normalize_target:
                target_torch, mean, std = normalize_instance(target_torch, eps=1e-11)
                target_torch = target_torch.clamp(-6, 6)

        else:
            target_torch = torch.Tensor([0])

        if self.provide_pseudinverse:
            if self.provide_measurement: ## DURING EVAL
                ## check for FLAIR 203
                crop_size = attrs['recon_size'][:-1]
                if image.shape[-2] < crop_size[1]:
                    crop_size = (image.shape[-3], image.shape[-2])
                if  self.target_type == "mvue":
                    target_torch = complex_abs(target_torch)
                    target_torch = center_crop(target_torch, crop_size)

                sens_maps = attrs["smaps"]
                output_mask = np.abs(np.sum(sens_maps * np.conj(sens_maps), axis=0, keepdims=True))
                output_mask = output_mask > 0.5
                output_mask = to_tensor(output_mask)
                attrs['output_mask'] = output_mask
                return masked_kspace, target_torch, image, attrs
            else:
                return target_torch, image
                
        else:
            if self.provide_measurement:
                return masked_kspace, target_torch
            else: ## THIS
                return target_torch
                # # Train on patches
                # size = 128
                # cropped = torch.zeros(size,size,2)
                # while cropped.norm() == 0:
                #     cropped = random_aug_and_crop(target_torch, size)
                # return cropped