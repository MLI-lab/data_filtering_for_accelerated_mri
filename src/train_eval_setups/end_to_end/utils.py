import fastmri
from fastmri.data import transforms, subsample
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
from .models import NormUnet, VisionTransformer, ReconNet, VarNet
import random
from math import ceil
import os
import time
import re
import json
import yaml

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def logging(output_dir, text):
    with open(os.path.join(output_dir, 'logs.txt'), 'a') as f:
        f.write(time.strftime('%X %x %Z')+', ')
        f.write(text+'\n')

def flatten(l):
    return [item for sublist in l for item in sublist]


def fft2_np(x):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1)) 

def ifft2_np(x):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1)) 

def rss_np(x, axis=0):
    return np.sqrt(np.sum(np.square(np.abs(x)), axis=axis))

def center_crop(data, shape):
    """
    Adjusted code from fastMRI github repository
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape
    Returns:
        The center cropped image.
    """
    assert shape[0] > 0 and shape[1] > 0, "Desired shape of ({:d},{:d}) invalid. shape must contain positive integers".format(*shape)
    w_from = (data.shape[-2] - shape[0]) // 2
    w_from = w_from if w_from > 0 else 0
    
    h_from = (data.shape[-1] - shape[1]) // 2
    h_from = h_from if h_from > 0 else 0

    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)

class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]
    
class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        path_to_json,
        challenge: str = 'multicoil',
        transform: Optional[Callable] = None,
        augment_data = False,
        load_sensitivity_maps = True,
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
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.augment_data = augment_data
        self.raw_samples = []
        self.path_to_json = path_to_json
        self.load_sensitivity_maps = load_sensitivity_maps
        

        with open(path_to_json) as f:
            json_data = json.load(f)

        for _, data in json_data['files'].items():
            file = Path(data['path'])
            metadata = self._retrieve_metadata(file)
            new_raw_samples = []
            for slice_ind in data['slices']:
                freq = data['slices'][slice_ind]['freq']
                slice_ind = int(slice_ind)
                for _ in range(freq):
                    raw_sample = FastMRIRawDataSample(file, slice_ind, metadata)
                    new_raw_samples.append(raw_sample)
                
            self.raw_samples += new_raw_samples

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            c, h, w = hf["kspace"].shape[-3], hf["kspace"].shape[-2], hf["kspace"].shape[-1]    
            ht, wt = hf[self.recons_key].shape[-2], hf[self.recons_key].shape[-1]
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
                    padding_right = hf["kspace"].shape[-1]
                enc_size = (h, w, 1) 
                recon_size = enc_size
                            
            padding_left = 0 if padding_left < 0 else padding_left
            padding_right = w if padding_right > w else padding_right

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                "target_shape": (ht, wt),
                "kspace_shape": (c, h, w),
                **hf.attrs,
            }

        return metadata

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.raw_samples[i]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            # target = hf[self.recons_key][dataslice] # rss
            target = hf['reconstruction_mvue'][dataslice] # mvue
            # center crop to intended recontruction shape
            target = transforms.center_crop(target, metadata['target_shape'])
            attrs = dict(hf.attrs)
            attrs.update(metadata)            
            h = target.shape[-2] if attrs["recon_size"][0] > target.shape[-2] else attrs["recon_size"][0]
            w = target.shape[-1] if attrs["recon_size"][1] > target.shape[-1] else attrs["recon_size"][1]                        
            attrs["recon_size"] = (h, w, 1)
            # Update max value for mvue
            attrs['max'] = hf.attrs['max_mvue']
            attrs['sensitivity_maps'] = None
            if self.load_sensitivity_maps and 'sensitivity_maps' in hf.keys():
                attrs['sensitivity_maps'] = hf['sensitivity_maps'][dataslice] 

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

class BatchSamplerSameShape(Sampler):
    r"""Yield a mini-batch of indices. The sampler will drop the last batch of
            an image size bin if it is not equal to ``batch_size``

    Args:
        examples (dict): List from dataset class.
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, dataset, batch_size, indices=None, shuffle=False, group_shape_by='target'):
        self.batch_size = batch_size
        self.data = {}
        self.shuffle = shuffle
        self.indices = range(len(dataset.raw_samples)) if indices is None else indices

        for idx in self.indices:
            item = dataset.raw_samples[idx]
            if group_shape_by == 'target':
                shape = item.metadata['target_shape']
            elif group_shape_by == 'kspace_and_target':
                shape = item.metadata['kspace_shape'] + item.metadata['target_shape']
            else:
                raise NotImplementedError(f'group_shape_by = \'{group_shape_by}\' not implemented')

            if shape in self.data:
                self.data[shape].append(idx)
            else:
                self.data[shape] = [idx]

        self.total = 0
        for shape, indices in self.data.items():
            self.total += ceil(len(indices) / self.batch_size)
            
    def __iter__(self):
        batches = []

        for _, indices in self.data.items():
            if self.shuffle:
                random.shuffle(indices)                
            batch = []
            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    batches.append(batch)
                    batch = []
            if batch:
                batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return self.total
    
class DistributedBatchSamplerSameShape(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True, batch_size = 1, group_shape_by='target') -> None:
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed,
                         drop_last=drop_last)
        self.batch_size = batch_size
        self.group_shape_by = group_shape_by

    def __iter__(self):
        indices = list(super().__iter__())
        batch_sampler = BatchSamplerSameShape(self.dataset, batch_size=self.batch_size, indices=indices, shuffle=self.shuffle, group_shape_by=self.group_shape_by)
        return iter(batch_sampler)

    def __len__(self) -> int:
        return self.num_samples//self.batch_size # lower bound

class VarNetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """

    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]
    output_mask: torch.Tensor
    sens_maps: torch.Tensor


class VarNetDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> VarNetSample:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A VarNetSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """
        if target is not None:
            # take absolute value of mvue reconstruction
            target = np.abs(target)
            target_torch = transforms.to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = transforms.to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])
        

        masked_kspace, mask_torch, num_low_frequencies = transforms.apply_mask(
            kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

        if attrs['sensitivity_maps'] is not None:
            sens_maps = attrs['sensitivity_maps']
            output_mask = np.abs(np.sum(sens_maps * np.conj(sens_maps), axis=0, keepdims=True))
            output_mask = output_mask > 0.5
            output_mask = transforms.to_tensor(output_mask)
            sens_maps = transforms.to_tensor(sens_maps)
        else:
            output_mask = torch.zeros_like(masked_kspace[:1,...,0]).bool()
            sens_maps = torch.zeros_like(masked_kspace)
            
        sample = VarNetSample(
            masked_kspace=masked_kspace,
            mask=mask_torch.to(torch.bool),
            num_low_frequencies=num_low_frequencies,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size,
            output_mask = output_mask,
            sens_maps = sens_maps,
        )
 
        return sample
    
class UnetSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.
    Args:
        image: Subsampled image after inverse FFT.
        target: The target image (if applicable).
        mean: Per-channel mean values used for normalization.
        std: Per-channel standard deviations used for normalization.
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
    """

    image: torch.Tensor
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    output_mask: torch.Tensor
    
class UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.
        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the filename, and the slice number.

        """
        kspace_torch = transforms.to_tensor(kspace)

        # check for max value
        max_value = attrs["max"]

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            masked_kspace = transforms.apply_mask(kspace_torch, self.mask_func, seed=seed)[0]
        else:
            masked_kspace = kspace_torch

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[-1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = transforms.complex_center_crop(image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = fastmri.rss(image, -3).unsqueeze(0)

        # normalize target
        if target is not None:
            # take absolute value of mvue reconstruction
            target = np.abs(target)
            target_torch = transforms.to_tensor(target)
        else:
            target_torch = torch.Tensor([0])
            
        
        if attrs['sensitivity_maps'] is not None:
            sens_maps = attrs['sensitivity_maps']
            output_mask = np.abs(np.sum(sens_maps * np.conj(sens_maps), axis=0, keepdims=True))
            output_mask = output_mask > 0.5
            output_mask = transforms.to_tensor(output_mask)
        else:
            output_mask = torch.zeros_like(image).bool()
        
        
        return UnetSample(
            image=image,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            output_mask = output_mask,
        )

def get_dataloader(path_to_json, transform, num_workers=4, augment_data=False, batch_size=1, shuffle=True, rank=0, world_size=1, group_shape_by=None, load_sensitivity_maps=True):
    """Get dataloader from list of fastmri files and transform"""

    dataset = SliceDataset(path_to_json, transform=transform, augment_data=augment_data, load_sensitivity_maps=load_sensitivity_maps)

    if group_shape_by is None:
        # backwards comp.
        if isinstance(transform, UnetDataTransform) or transform is None:
            group_shape_by = 'target'
        elif isinstance(transform, VarNetDataTransform):
            group_shape_by = 'kspace_and_target'

    if world_size > 1:
        sampler = DistributedBatchSamplerSameShape(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, batch_size=batch_size, group_shape_by=group_shape_by)
    else:
        sampler = BatchSamplerSameShape(dataset, shuffle=shuffle, batch_size=batch_size, group_shape_by=group_shape_by)

    dataloader = DataLoader(dataset, pin_memory=False, num_workers=num_workers, batch_sampler=sampler)
    
    return dataloader

def unet_forward(model, sample, device):
    input = sample.image.to(device)
    output = model(input)

    return output

def varnet_forward(model, sample, device):
    input = sample.masked_kspace.to(device)
    mask = sample.mask.to(device)
    if not model.training:
        sens_maps = sample.sens_maps.to(device)
    else:
        sens_maps = None
    output = model(input, mask, sens_maps_eval=sens_maps).unsqueeze(-3)
    return output


def mask_func_params(accl_factors):
    center_fractions = []
    accelerations = []
    for accl_factor in accl_factors:
        if accl_factor == 2:
            center_fractions.append(0.16)
            accelerations.append(2)
        elif accl_factor == 3:        
            center_fractions.append(0.12)
            accelerations.append(3)
        elif accl_factor == 4:        
            center_fractions.append(0.08)
            accelerations.append(4)
        elif accl_factor == 6:
            center_fractions.append(0.06)
            accelerations.append(6)
        elif accl_factor == 7:
            center_fractions.append(0.05)
            accelerations.append(7)
        elif accl_factor == 8:
            center_fractions.append(0.04)
            accelerations.append(8)
        elif accl_factor == 16:
            center_fractions.append(0.02)
            accelerations.append(16)
    return center_fractions, accelerations


def setup(mode, config, accl_factors, seed=0):
    torch.manual_seed(seed)

    if mode == 'train':
        use_seed = False

    elif mode == 'test':
        use_seed = True

    center_fractions, accelerations = mask_func_params(accl_factors)
    
    mask_func = subsample.EquiSpacedMaskFunc(
        center_fractions=center_fractions,
        accelerations=accelerations,
        )

    #model_name = config['model']
    #hyperparams = config['hyperparams']

    model_name = config['model']
    hyperparams = config['hyperparams']
        
    assert model_name in ['u-net', 'vit', 'varnet'], f'\'{model_name}\' is not implemented.' 
        
    if model_name == 'u-net':
        model = NormUnet(**hyperparams)
        data_transform = UnetDataTransform('multicoil', mask_func, use_seed=use_seed)
        forward_fn = unet_forward

    elif model_name == 'vit':
        net = VisionTransformer(**hyperparams)
        model = ReconNet(net)
        data_transform = UnetDataTransform('multicoil', mask_func, use_seed=use_seed)
        forward_fn = unet_forward
    
    elif model_name == 'varnet':
        model = VarNet(**hyperparams)
        data_transform = VarNetDataTransform(mask_func, use_seed=use_seed)
        forward_fn = varnet_forward

    return model, data_transform, forward_fn