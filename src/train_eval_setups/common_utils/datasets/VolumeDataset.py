import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from warnings import warn

import h5py
import numpy as np
import pandas as pd
import requests
import torch
import yaml
from tqdm import tqdm

from ...diff_model_recon.physics_trafos.trafo_datasets.fftn3d import ifftshift, fftshift

from fastmri.data.mri_data import et_query
from ....interfaces.dataset_models import DatasetModel

class FastMRIRawVolumeSample(NamedTuple):
    fname: Path
    slice_count: int
    metadata: Dict[str, Any]

class VolumeDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        dataset_model : DatasetModel,
        challenge: str = 'multicoil',
        transform: Optional[Callable] = None,
        readout_dim_fftshift_cor : Tuple[bool] = [True, True], # (True, True) for CC359, (True, False) for Stanford 3D
    ):

        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        # we assume the structure of the volumes is (num_slices, num_coils, height, width, 2)
        self.readout_dim_fftshift_cor = readout_dim_fftshift_cor
        self.transform = transform

        self.raw_samples = []
        for volume_model in tqdm(dataset_model.files.values(), desc="Loading volumes and retrieving metadata."):
            file_path = Path(volume_model.path)
            metadata, num_slices = self._retrieve_metadata(file_path)
            raw_sample = FastMRIRawVolumeSample(file_path, num_slices, metadata)
            self.raw_samples.append(raw_sample)

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:

            if "ismrmrd_header" in hf:
                et_root = etree.fromstring(hf["ismrmrd_header"][()])

                enc = ["encoding", "encodedSpace", "matrixSize"]
                enc_size = (
                    int(et_query(et_root, enc + ["x"])),
                    int(et_query(et_root, enc + ["y"])),
                    int(et_query(et_root, enc + ["z"])),
                )
                rec = ["encoding", "reconSpace", "matrixSize"]
                recon_size = (
                    int(et_query(et_root, rec + ["x"])),
                    int(et_query(et_root, rec + ["y"])),
                    int(et_query(et_root, rec + ["z"])),
                )

                lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
                enc_limits_center = int(et_query(et_root, lims + ["center"]))
                enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

                padding_left = enc_size[1] // 2 - enc_limits_center
                padding_right = padding_left + enc_limits_max

                metadata_ismrmrd = {
                    "padding_left": padding_left,
                    "padding_right": padding_right,
                    "encoding_size": enc_size,
                    "recon_size": recon_size
                }
            else:
                #warn(f"No ISMRMRD header found in {fname}.")
                metadata_ismrmrd = {}

            extra_dict = {}
            if not "kspace_vol_norm" in hf.attrs:
                kspace = hf["kspace"][()]
                extra_dict = {
                    "kspace_vol_norm": np.linalg.norm(kspace)
                }
                kspace = None
                torch.cuda.empty_cache()

            num_slices = hf["kspace"].shape[0]
                
            metadata = {
                "num_slices" : num_slices,
                **hf.attrs,
                **extra_dict,
                **metadata_ismrmrd
            }

        return metadata, num_slices

    def __len__(self):
        return len(self.raw_samples)

    def _fft1c(self, data: torch.Tensor, norm: str = "ortho", dim:int = 0, shifts_enable : Tuple[bool] = (True, True)) -> torch.Tensor:
        # True, False -> Stanford 3D
        # True, True -> default
        if shifts_enable[0]:
            data = ifftshift(data, dim=[dim])
        data = torch.view_as_real(
            torch.fft.fftn(  # type: ignore
                torch.view_as_complex(data), dim=[dim], norm=norm
            )
        )
        if shifts_enable[1]:
            data = fftshift(data, dim=[dim])
        return data

    def _ifft1c(self, data: torch.Tensor, norm: str = "ortho", dim:int = 0) -> torch.Tensor:
        data = ifftshift(data, dim=[dim])
        data = torch.view_as_real(
            torch.fft.ifftn(  # type: ignore
                torch.view_as_complex(data), dim=[dim], norm=norm
            )
        )
        return fftshift(data, dim=[dim])

    def __getitem__(self, i: int):
        fname, slice_count, metadata = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            # this is pretty standard
            kspace = hf["kspace"][:] #[dataslice]
            kspace = torch.view_as_real(torch.from_numpy(kspace))
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf['reconstruction_mvue'][:] if 'reconstruction_mvue' in hf else None
            target = torch.view_as_real(torch.from_numpy(target))

            # shape is (slices in Z-dir, coils, Y, X, 2) -> (coils, Z, Y, X, 2)
            kspace = kspace.movedim(1, 0)

            # we assume that the first dim is spatial now (nr of slices = Z dir) -> transform into 3D kspace
            # for some datasets one needs to omit an fft shift on the readout dim, for others not
            kspace = self._fft1c(kspace, dim=1, norm="ortho", shifts_enable=self.readout_dim_fftshift_cor)

            attrs = dict(hf.attrs)
            attrs['max'] = hf.attrs['max_mvue']
            attrs['smaps'] = hf['sensitivity_maps'][:] if 'sensitivity_maps' in hf else None
            attrs.update(metadata)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, slice_count)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, slice_count)

        return sample