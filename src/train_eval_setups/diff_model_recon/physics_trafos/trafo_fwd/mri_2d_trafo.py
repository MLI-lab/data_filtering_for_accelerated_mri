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

class SubsampledFourierTrafo(BaseRayTrafo):
    """
    Subsampled Fourier ransform implemented by (sparse) matrix multiplication.

    """

    def __init__(self,
            zero_padding: Tuple[int, int],
            input_real2complex: bool,
            mask_enabled : bool,
            mask_type : str,
            mask_accelerations : Tuple[int],
            mask_center_fractions : Tuple[float],
            mask_use_seed : int,
            include_sensitivitymaps : bool,
            sensitivitymaps_complex : bool,
            sensitivitymaps_fillouter : bool,
            conv_averaging_shape : Optional[Tuple[int, int, int]],
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

        self.zero_padding = zero_padding
        if self.zero_padding is not None:
            self.pad_sizes = [0, 160]
            self.pad_trafo = T.Pad(self.pad_sizes)

        self.input_real2complex = input_real2complex

        self.mask_use_seed = mask_use_seed
        self.mask_enabled = mask_enabled
        self.mask_center_fractions = mask_center_fractions
        self.mask_accelerations = mask_accelerations
        self.mask_type = mask_type

        self.include_sensitivitymaps = include_sensitivitymaps
        self.sensitivitymaps_complex = sensitivitymaps_complex
        self.sensitivitymaps_fillouter = sensitivitymaps_fillouter
        self.conv_averaging_shape = conv_averaging_shape

    def _set_mask(self, obs_shape) -> None:
        if self.mask_enabled:
            mask_class = fastmri.data.subsample.RandomMaskFunc if self.mask_type == 'random' else fastmri.data.subsample.EquispacedMaskFractionFunc
            mask_func = mask_class(center_fractions=self.mask_center_fractions, accelerations=[self.mask_accelerations], seed=self.mask_use_seed)
            shape = (1,) * len(obs_shape[:-3]) + tuple(obs_shape[-3:])
            self.mask, num_low_frequencies = mask_func(shape, offset=None, seed=self.mask_use_seed)
        else:
            self.mask = None

    def calibrate(self, observation: Tensor, calib_params) -> None:

        if self.mask_enabled:
            if "mask" in calib_params:
                self.mask = calib_params["mask"]
            else:
                logging.warning("Mask not provided, generating new mask.")
                self._set_mask(observation.shape)

        if self.include_sensitivitymaps:
            if "smaps" in calib_params:
                logging.info("Using calibration parameters provided.")
                sens_maps = calib_params["smaps"].numpy()
            else:
                from src.train_eval_setups.diff_model_recon.utils.bart_utils import compute_sens_maps_mp
                logging.warning("No calibration parameters provided, computing sensitivity maps.")
                # assume shape is (coils, X, Y, 2)
                sens_maps = compute_sens_maps_mp(observation[None])

            S = sens_maps.copy() 
            S = np.stack((S.real, S.imag), axis=-1)

            if self.sensitivitymaps_fillouter:
                for coil_ind,s in enumerate(S):
                    ss = np.copy(s)
                    ss[np.abs(ss)==0.0] = np.abs(ss).max()
                    S[coil_ind,...] = ss
            S = torch.from_numpy(S).to(observation.device)

            if self.sensitivitymaps_complex:
                S = torch.view_as_complex(S)

            self.sense_matrix = S
            S_norm = torch.sum(S.abs().square(), dim=1).unsqueeze(dim=1).sqrt() # shape (1, 1, kx, ky)
            # crate a Tensor with ones where S_norm is not 0 and 0 otherwise
            #S_norm_binary = torch.where(S_norm == 0, torch.zeros_like(S_norm), torch.ones_like(S_norm))
            #torch.testing.assert_close(S_norm, S_norm_binary)
            return S_norm
        else:
            self.sense_matrix = None
            return None

    def trafo(self, x: Tensor) -> Tensor:
        # expected shape: (1, 320, 320, 1) or (1, 320, 320, 2)
        if self.zero_padding is not None:   
            # todo: this could surely be performed more elegantly
            bcwh = x.unsqueeze(-4).swapaxes(-4, -1).squeeze(-1)
            bcwh_p = self.pad_trafo(bcwh)
            x = bcwh_p.unsqueeze(-1).swapaxes(-1, -4).squeeze(-4)

        if self.input_real2complex:
            #x = torch.stack( [x, torch.zeros_like(x)] , dim=-1)
            x = F.pad(x, (0, 1, 0, 0), "constant", 0)

        if self.conv_averaging_shape is not None:
            # (1, 210, 640, 320, 2) -> (1, 35, 320, 320)
            assert len(x.shape) == 5
            B, Z, kX, kY, C = x.shape
            x = x.moveaxis(-1, 0).view(C*B, 1, Z, kX, kY) # add color dim and move complex into batch
            kernel = torch.ones( (1,1) + self.conv_averaging_shape, device=x.get_device()) / torch.prod( torch.Tensor(self.conv_averaging_shape) )
            x = F.conv3d(x,
                weight=kernel,
                stride=self.conv_averaging_shape).view(C, B, -1, kX, kY).moveaxis(0,-1)

        if self.include_sensitivitymaps:
            x = x.unsqueeze(-4) # add coil dim
            if self.sensitivitymaps_complex:
                x = torch.view_as_real(
                    torch.view_as_complex(x) * self.sense_matrix.to(x.get_device())
                )
            else:
                x = x * self.sense_matrix.to(x.get_device())

        y = fastmri.fft2c(x)
        if self.mask is not None:
            mask = self.mask.to(y.get_device())
            mask.requires_grad = True
            return y * mask
        else:
            return y

    def trafo_adjoint(self, y: Tensor) -> Tensor:
        # x^hat = F^T M^T y
        x_hat = fastmri.ifft2c(y) # (1, 640, 320, 2)

        if self.include_sensitivitymaps:
            if self.sensitivitymaps_complex:
                # (1, 256, 15, 320, 320, 2) in real, (1, 256, 15, 320, 320) complex
                # sensmatrix (15, 1, 320, 320)
                # for sensmatrix * data in forward,
                # - coil dim is added
                # - data broadcasted along dimension
                # - elemntwise multiplication
                x_hat = torch.view_as_real(
                    torch.sum(
                        torch.conj(self.sense_matrix).to(x_hat.get_device()) * torch.view_as_complex(x_hat)
                    , dim=-3) # sum if one sense map for all slices
                )
            else:
                x_hat = torch.sum(self.sense_matrix.to(x_hat.get_device()) * x_hat, dim=1)


        if self.conv_averaging_shape is not None:
            for dim, rep in enumerate(self.conv_averaging_shape):
                x_hat = x_hat.repeat_interleave(repeats=rep, dim=dim)

        if self.input_real2complex:
            #x_hat = fastmri.complex_abs(x_hat) # (1, 320, 320)
            x_hat = x_hat[..., 0].unsqueeze(-1)

        if self.zero_padding is not None:
            crop_size = (320, 320)
            x_hat = fastmri.data.transforms.complex_center_crop(x_hat, crop_size) # (1, 320, 320, 2)

        return x_hat

    trafo_flat = BaseRayTrafo._trafo_flat_via_trafo
    trafo_adjoint_flat = BaseRayTrafo._trafo_adjoint_flat_via_trafo_adjoint
