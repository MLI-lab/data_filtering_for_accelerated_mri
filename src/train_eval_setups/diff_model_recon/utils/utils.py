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

        data_provides_pseudoinverse = False
        data_provides_measurement = False

        #self.indices = range(len(dataset.raw_samples)) if indices is None else indices
        self.indices = range(len(dataset)) if indices is None else indices

        for idx in self.indices:

            try:
                item = dataset.raw_samples[idx]
                if group_shape_by == 'target':
                    shape = item.metadata['target_shape']
                elif group_shape_by == 'kspace_and_target':
                    shape = item.metadata['kspace_shape'] + item.metadata['target_shape']
                else:
                    raise NotImplementedError(f'group_shape_by = \'{group_shape_by}\' not implemented')
            except: 
                item = dataset[idx] # either contains kspace, target as first two, or only target
                target_shape = item.shape if not data_provides_pseudoinverse and not data_provides_measurement else item[1].shape
                obs_shape = item[0].shape if data_provides_measurement else None
                if group_shape_by == 'target':
                    shape = target_shape
                elif group_shape_by == 'kspace_and_target':
                    shape = obs_shape + target_shape
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

#def get_dataloader(path_to_json, transform, num_workers=4, augment_data=False, batch_size=1, shuffle=True, rank=0, world_size=1, group_shape_by=None):
    #"""Get dataloader from list of fastmri files and transform"""

    #dataset = SliceDataset(path_to_json, transform=transform, augment_data=augment_data)
def get_dataloader(dataset, num_workers=4, batch_size=1, shuffle=True, rank=0, world_size=1, group_shape_by=None):
    if group_shape_by is None:
        # during eval
        return DataLoader(dataset, pin_memory=False, num_workers=num_workers)
    else:
        # during train
        if world_size > 1:
            sampler = DistributedBatchSamplerSameShape(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, batch_size=batch_size, group_shape_by=group_shape_by)
        else:
            # return DataLoader(dataset, pin_memory=False, batch_size=batch_size,num_workers=num_workers, shuffle=True) # when training with patches
            sampler = BatchSamplerSameShape(dataset, shuffle=shuffle, batch_size=batch_size, group_shape_by=group_shape_by)

        return DataLoader(dataset, pin_memory=False, num_workers=num_workers, batch_sampler=sampler)

def unet_forward(model, sample, device):
    input = sample.image.to(device)
    output = model(input)

    return output

def varnet_forward(model, sample, device):
    input = sample.masked_kspace.to(device)
    mask = sample.mask.to(device)
    output = model(input, mask).unsqueeze(-3)
    
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
        elif accl_factor == 8:
            center_fractions.append(0.04)
            accelerations.append(8)
        elif accl_factor == 16:
            center_fractions.append(0.02)
            accelerations.append(16)
    return center_fractions, accelerations


#def setup(mode, config, accl_factors, seed=0):
    #torch.manual_seed(seed)

    #if mode == 'train':
        #use_seed = False

    #elif mode == 'test':
        #use_seed = True

    #center_fractions, accelerations = mask_func_params(accl_factors)

    #mask_func = subsample.EquiSpacedMaskFunc(
        #center_fractions=center_fractions,
        #accelerations=accelerations,
        #)

    ##model_name = config['model']
    ##hyperparams = config['hyperparams']

    #model_name = config['model']
    #hyperparams = config['hyperparams']
        
    ##assert model_name in ['u-net', 'vit', 'varnet'], f'\'{model_name}\' is not implemented.' 
        
    ##if model_name == 'u-net':
        ##model = NormUnet(**hyperparams)
        ##dataset_trafo = UnetDataTransform('multicoil', mask_func, use_seed=use_seed)
        ##forward_fn = unet_forward

    ##elif model_name == 'vit':
        ##net = VisionTransformer(**hyperparams)
        ##model = ReconNet(net)
        ##dataset_trafo = UnetDataTransform('multicoil', mask_func, use_seed=use_seed)
        ##forward_fn = unet_forward
    
    ##elif model_name == 'varnet':
        ##model = VarNet(**hyperparams)
        ##dataset_trafo = VarNetDataTransform(mask_func, use_seed=use_seed)
        ##forward_fn = varnet_forward

    

    #return model, dataset_trafo, forward_fn

from typing import Tuple
from torch import Tensor
import os


def get_path_by_cluster_name(param, cfg):
    cluster_name = os.environ.get("CLUSTER_NAME", cfg.cluster_name)
    # Some parameters are not used and set to None
    if param is None:
        return None
    elif cluster_name in param:
        return param[cluster_name]
    elif "default" in param:
        return param["default"]
    else:
        raise ValueError(f"Cluster name {cluster_name} not found in {param}.")

def midslice2selct(x: Tensor) -> Tensor:
    if x.dim() < 5: 
        return x 
    elif x.dim() == 5: 
        return x[:, :, x.shape[2] // 2, ...]
    #elif x.dim() == 4 and x.shape[0] == 1:
        #return x[0, ...]
    else: 
        raise ValueError

def normalize_clamp(im, cl=3, cu=3):
    im_std = im.std()
    im_mean = im.mean()
    im = (im-im_mean) / im_std
    im = im.clamp(-cl,cu)
    return im

def align_normalization(im_to,im_from, c=6):
    # im1: ground truth
    # im2: reconstruction
    im_to = (im_to-im_to.mean()) / im_to.std()
    im_to = im_to.clamp(-c,c)

    im2_std = im_from.std()
    im2_mean = im_from.mean()

    im_from = (im_from-im_from.mean()) / im_from.std()
    im_from = im_from.clamp(-c,c)

    im_to *= im2_std
    im_to += im2_mean

    im_from *= im2_std
    im_from += im2_mean

    return im_to,im_from

# %%
import torch
import torch.nn.functional as F

def build_grid(source_size,target_size, device):

    target_size_y, target_size_x = target_size
    source_size_y, source_size_x = source_size

    k_y = float(target_size_y)/float(source_size_y) # 80 / 320 = 0.25
    direct_y = torch.linspace(-1, 2*k_y - 1,target_size_y, device=device).unsqueeze(0).repeat(target_size_y,1).unsqueeze(-1)

    k_x = float(target_size_x)/float(source_size_x) # 80 / 320 = 0.25
    direct_x = torch.linspace(-1, 2*k_x - 1,target_size_x, device=device).unsqueeze(0).repeat(target_size_x,1).unsqueeze(-1)

    return torch.cat([direct_y, direct_x.transpose(1,0)],dim=2).unsqueeze(0)

def random_crop_grid(x,grid, slab_thickness):
    delta_y = x.size(2)-grid.size(1) # 320 - 80 = 240
    delta_x = x.size(3)-grid.size(2) # 320 - 80 = 240

    grid = grid.repeat(x.size(0),1,1,1) # repeat grid over batch size

    add_y = ((torch.rand(size=(x.size(0) // slab_thickness, 1, 1), device=x.device) * delta_y).expand(-1, grid.size(1), grid.size(2)) / x.size(2)).repeat_interleave(slab_thickness, dim=0)
    #grid[:,:,:,0] = grid[:,:,:,0] + 2*(torch.rand(size=(x.size(0), 1, 1), device=x.device) * delta_y).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
    grid[:,:,:,0] = grid[:,:,:,0] + 2*add_y
    add_x = ((torch.rand(size=(x.size(0) // slab_thickness, 1, 1), device=x.device) * delta_x).expand(-1, grid.size(1), grid.size(2)) / x.size(3)).repeat_interleave(slab_thickness, dim=0)
    grid[:,:,:,1] = grid[:,:,:,1] + 2*add_x
    return grid

def crop_random_over_batches(batch, crop_size : Tuple[int, int] = (80, 80), slab_thickness : int = 1):
    grid_source = build_grid(batch.shape[-2:], crop_size, batch.device)
    grid_shifted = random_crop_grid(batch,grid_source, slab_thickness)
    return F.grid_sample(batch, grid_shifted, mode="nearest")