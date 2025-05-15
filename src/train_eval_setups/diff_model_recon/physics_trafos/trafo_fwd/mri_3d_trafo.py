"""
Provides :class:`MatmulRayTrafo`.
"""

from __future__ import annotations  # postponed evaluation, to make ArrayLike look good in docs
from typing import Union, Optional, Callable, Tuple, Any
try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any
import torch
from torch import Tensor
import numpy as np
import scipy.sparse
import logging

from .base_trafo_fwd import BaseRayTrafo
import fastmri
import fastmri.data.subsample
import fastmri.data.transforms
import torch.nn.functional as F
import torchvision.transforms as T

from ..trafo_datasets.fftn3d import fft3c, ifft3c
from ..trafo_datasets.transforms import apply_mask

from sigpy.mri.samp import poisson, radial, spiral

class SubsampledFourierTrafo3D(BaseRayTrafo):
    """
    Subsampled Fourier ransform implemented by (sparse) matrix multiplication.

    """

    def __init__(self,
            mask_enabled : bool,
            mask_type : str,
            mask_accelerations : Tuple[int],
            mask_center_fractions : Tuple[float],
            mask_use_seed : int,
            include_sensitivitymaps : bool,
            sensitivitymaps_complex : bool,
            sensitivitymaps_fillouter : bool,
            wrapped_2d_mode : bool
        ):
            # tbd: add masking
        """
        Parameters
        ----------
        im_shape, obs_shape
            See :meth:`BaseRayTrafo.__init__`.
        fbp_fun : callable, optional
            Function applying a filtered back-projection, used for providing
            :meth:`fbp`.
        """
        super().__init__()

        self.mask_seed = mask_use_seed
        self.mask_enabled = mask_enabled
        self.mask_center_fractions = mask_center_fractions
        self.mask_accelerations = mask_accelerations
        self.mask_type = mask_type
        self.wrapped_2d_mode = wrapped_2d_mode

        #self._set_mask(obs_shape)

        self.include_sensitivitymaps = include_sensitivitymaps
        self.sensitivitymaps_complex = sensitivitymaps_complex
        self.sensitivitymaps_fillouter = sensitivitymaps_fillouter


    def _set_mask(self, obs_shape, calib_params) -> None:
        if self.mask_enabled:
            if self.mask_type == 'dataset':
                assert "mask" in calib_params, "mask not found in calibration parameters"
                self.mask = calib_params["mask"]
            elif self.mask_type == 'Poisson2D':
                #self.mask = torch.from_numpy(poisson([320, 320], self.mask_accelerations[0], seed=self.mask_seed).astype(np.float32)).unsqueeze(dim=-1)
                self.mask = torch.from_numpy(poisson(obs_shape[-3:-1], self.mask_accelerations, seed=self.mask_seed).astype(np.float32)).unsqueeze(dim=-1)
            else:
                mask_class = fastmri.data.subsample.RandomMaskFunc if self.mask_type == 'random' else fastmri.data.subsample.EquispacedMaskFractionFunc
                mask_func = mask_class(center_fractions=self.mask_center_fractions, accelerations=[self.mask_accelerations], seed=self.mask_seed)
                shape = (1,) * len(obs_shape[:-3]) + tuple(obs_shape[-3:])
                self.mask, num_low_frequencies = mask_func(shape, offset=None, seed=self.mask_seed)
        else:
            self.mask = None

    def calibrate(self, observation: Tensor, calib_params) -> None:

        if self.mask_enabled:
            if "mask" in calib_params:
                self.mask = calib_params["mask"].unsqueeze(dim=-1).type(torch.float32)
            else:
                logging.warning("Mask not provided, generating new mask.")
                self._set_mask(observation.shape)

        if self.include_sensitivitymaps:
            
            if "smaps" in calib_params:
                logging.info("Using calibration parameters provided") # ??? unclear
                #if calib_params["smaps"].shape[0] != 1:
                    #S = torch.view_as_real(
                        #torch.from_numpy(calib_params["smaps"]).moveaxis(-1,0)
                    #)
                #else:
                    #S = torch.view_as_real(
                            #calib_params["smaps"].squeeze(0).moveaxis(-1,0)
                        #)

                sens_maps = calib_params["smaps"].numpy()
                sens_maps = np.moveaxis(sens_maps[0], 1, 0)
            else:
                from src.train_eval_setups.diff_model_recon.utils.bart_utils import compute_sens_maps_3d
                sens_maps = compute_sens_maps_3d(observation[0])
                sense_maps = np.moveaxis(sense_maps, -1, 0)

            S = sens_maps.copy() 
            S = np.stack((S.real, S.imag), axis=-1)
            S = torch.from_numpy(S)

            if self.sensitivitymaps_fillouter:
                for coil_ind,s in enumerate(S):
                    ss = np.copy(s)
                    ss[np.abs(ss)==0.0] = np.abs(ss).max()
                    S[coil_ind,...] = ss

            if self.sensitivitymaps_complex:
                S = torch.view_as_complex(S)
            else:
                S = torch.view_as_real(S)

            #for coil in range(S.shape[0]):
                #S[coil][S[coil] == 0.0] = S[coil][S[coil] != 0.0].abs().max()

            self.sense_matrix = S.to(observation.get_device()).type(torch.complex32)
            S = None

            torch.cuda.empty_cache()

        else:
            self.sense_matrix = None

    def trafo(self, x: Tensor, slice_inds : Optional[Tensor] = None, slice_axis : Optional[int]= None) -> Tensor:

        #if self.include_sensitivitymaps and self.sensitivitymaps_complex and self.wrapped_2d_mode and self.mask is not None and slice_inds is not None:
            #return fastmri.fft2c(
                    #torch.view_as_real(
                        #torch.view_as_complex(x.unsqueeze(-5)) * self.sense_matrix.index_select(slice_axis, slice_inds)
                    #)
            #) * self.mask
        #else:
            # assumed shape is (1, Z, X, Y, 2)
        if self.include_sensitivitymaps:
            S = self.sense_matrix.to(x.get_device())
            if slice_inds is not None and slice_axis is not None:
                S = S.index_select(slice_axis+1, slice_inds) # add +1 to skip coil dim in f ront
            # x shape (1, Coils, Z, Y, X, 2)
            if self.sensitivitymaps_complex:
                x = torch.view_as_real(
                    torch.view_as_complex(x.unsqueeze(-5)) * S
                )
            else:
                x = x.unsqueeze(-5) * S

        y = fft3c(x) if not self.wrapped_2d_mode else fastmri.fft2c(x)
        if self.mask is not None:
            return y * self.mask.to(y.get_device()) + 0.0
        else:
            return y

    def trafo_adjoint(self, y: Tensor) -> Tensor:
        # x^hat = F^T M^T y
        # assumed shape is (1, Coils, kZ, kY, kX, 2)
        # example (1, 8, 256, 320, 320, 2)
        x_hat = ifft3c(y) if not self.wrapped_2d_mode else fastmri.ifft2c(y)

        if self.include_sensitivitymaps:
            if self.sensitivitymaps_complex:
                x_hat = torch.view_as_real(
                    torch.sum(
                        torch.conj(self.sense_matrix).to(x_hat.get_device()) * torch.view_as_complex(x_hat)
                    , dim=-4)
                )
            else:
                x_hat = torch.sum(self.sense_matrix.to(x_hat.get_device()) * x_hat, dim=1)

        return x_hat

    trafo_flat = BaseRayTrafo._trafo_flat_via_trafo
    trafo_adjoint_flat = BaseRayTrafo._trafo_adjoint_flat_via_trafo_adjoint
