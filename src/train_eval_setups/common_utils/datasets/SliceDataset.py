import fastmri
from fastmri.data import transforms, subsample
from fastmri.data.transforms import VarNetDataTransform
from fastmri.data.subsample import MaskFunc
import h5py
import numpy as np
import torch
from pathlib import Path
import xml.etree.ElementTree as etree

from scipy.ndimage import fourier_shift

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

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
#from .models import NormUnet, VisionTransformer, ReconNet, VarNet
import random
from math import ceil
import os
import time
import re
import json
import yaml
from tqdm import tqdm

from src.train_eval_setups.diff_model_recon.utils.utils import FastMRIRawDataSample, et_query

from ....interfaces.dataset_models import DatasetModel, DatasetSliceModel, DatasetVolumeModel

class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        dataset_model : DatasetModel,
        challenge: str = 'multicoil',
        transform: Optional[Callable] = None,
        augment_data = False,
    ):
        """
        Args:
            path_to_json: path to a JSON file containing filepaths to the volumes and slices to be used
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
        """
        
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        # Best would be to set recons_key as reconstruction_mvue here already
        # The reason why reconstructin_rss is still picked is that for evaluation we crop to the rss ground truth shape as this is the area of interest
        # But during training we not cropping the mvue is a reasonable apporach so the diffusion model can generate full sized images for forward map reasons
        # We can also try training on cropped images and see how this performs (as an anecdote: the loss for e2e varnet is also computed only on the cropped images)
        # Probably cleaner to simply store crop size as an attribute in the h5 files
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.augment_data = augment_data
        self.raw_samples = []

        for volume_model in tqdm(dataset_model.files.values(), desc="Loading volumes and retrieving metadata"):
            file = Path(volume_model.path)
            metadata = self._retrieve_metadata(file)
            new_raw_samples = []
            for slice_ind in volume_model.slices:
                freq = volume_model.slices[slice_ind].freq
                slice_ind = int(slice_ind)
                for _ in range(freq):
                    raw_sample = FastMRIRawDataSample(file, slice_ind, metadata)
                    new_raw_samples.append(raw_sample)
                
            self.raw_samples += new_raw_samples

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if 'kspace' in hf.keys():
                c, h, w = hf["kspace"].shape[-3], hf["kspace"].shape[-2], hf["kspace"].shape[-1]
            if "reconstruction_rss" in hf.keys():
                _, hs, ws = hf["reconstruction_rss"].shape
            else:
                hs, ws = hf.attrs['rss_shape']
                
            # mvue shape
            hm, wm = hf['reconstruction_mvue'].shape[-2], hf['reconstruction_mvue'].shape[-1]
            h, w = hm, wm
            if "ismrmrd_header" in list(hf.keys()):
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
            else:
                attrs = dict(hf.attrs)
                if 'padding_left' in attrs:
                    padding_left = attrs['padding_left']
                else:
                    padding_left = 0
                    
                if 'padding_right' in attrs:
                    padding_right = attrs['padding_right']
                else:
                    if 'kspace' in hf.keys():
                        padding_right = hf["kspace"].shape[-1]
                    else:
                        padding_right = w
                enc_size = (h, w, 1) 
                recon_size = (hs, ws, 1) # RSS reconstruction shape
                            
            padding_left = 0 if padding_left < 0 else padding_left
            padding_right = w if padding_right > w else padding_right

            extra_dict = {}
            # if not "kspace_vol_norm" in hf.attrs:
            #     kspace = hf["kspace"][()]
            #     extra_dict = {
            #         "kspace_vol_norm": np.linalg.norm(kspace)
            #     }
            #     kspace = None
            #     torch.cuda.empty_cache()

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                "target_shape": (hm, wm),
                **hf.attrs,
                **extra_dict
            }
            if 'kspace' in hf.keys():
                metadata["kspace_shape"] = (c, h, w)

        return metadata

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.raw_samples[i]
        with h5py.File(fname, "r") as hf:
            if 'kspace' in hf.keys():
                kspace = hf["kspace"][dataslice]
            else:
                kspace = None
            # kspace = None # faster dataloading during train
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            # target = hf[self.recons_key][dataslice] if self.recons_key in hf else None
            # load mvue reconstruction
            target = hf['reconstruction_mvue'][dataslice]
            attrs = dict(hf.attrs)
            attrs.update(metadata)
            # Update max value for mvue
            attrs['max'] = hf.attrs['max_mvue']
            attrs['smaps'] = hf['sensitivity_maps'][dataslice] if 'sensitivity_maps' in hf.keys() else None
            # attrs['smaps'] = None # faster dataloading during train
            attrs['kspace_norm'] = hf.attrs['kspace_norm'][dataslice] if 'kspace_norm' in list(hf.attrs) else np.nan
            

            if self.augment_data:
                if np.random.rand() > 0.5: # vertical flip               
                    target = np.flip(target, axis=-2)
                    kspace = np.flip(kspace, axis=-2)
                    kspace = fourier_shift(kspace, (0,0,-1,0))
                    
                if np.random.rand() > 0.5: # horizontal flip
                    target = np.flip(target, axis=-1)
                    kspace = np.flip(kspace, axis=-1)
                    kspace = fourier_shift(kspace, (0,0,0,-1))
                    pad_left = attrs["padding_left"]
                    attrs["padding_left"] = target.shape[-1] - attrs["padding_right"]
                    attrs["padding_right"] = target.shape[-1] - pad_left

                if np.random.rand() > 0.5: # 90 degree rotation
                    target = np.rot90(target, axes=(-2, -1))
                    kspace = np.rot90(kspace, axes=(-2, -1))
                    kspace = fourier_shift(kspace, (0,0,-1,0))
                    attrs["padding_left"] = 0
                    attrs["padding_right"] = target.shape[-1]
                    h, w, z = attrs["encoding_size"]
                    attrs["encoding_size"] = (w, h, z)
                    h, w, z = attrs["recon_size"]
                    attrs["recon_size"] = (w, h, z)
                    
                target = np.ascontiguousarray(target)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, dataslice)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, dataslice)
                
        return sample