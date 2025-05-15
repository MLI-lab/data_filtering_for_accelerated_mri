import torch
from typing import Tuple, List, Dict
import math
from abc import ABC, abstractmethod


class SliceMethod(ABC):
    @abstractmethod
    def __call__(self, volume : torch.Tensor, outer_iteration : int, inner_iteration : int) -> List[torch.Tensor]:
        pass

class RandomSlabsMethod(SliceMethod):
    def __init__(self, slice_budget: int = 5, slab_thickness : int = 3, slice_stride : int = 0,
                volume_indices : Tuple[int, int, int] = [2, 3, 4], # Z, Y, X
                slice_enabled : Tuple[bool, bool, bool] = [True, True, True],
                swapaxis : Tuple[bool, bool, bool] = [False, False, False],
                rnd_indices : Tuple[bool, bool, bool] = [False, False, False],
                keep_dims : Tuple[bool, bool, bool] = [False, False, False]):

        self.slice_budget = slice_budget
        self.slab_thickness = slab_thickness
        self.stride = slice_stride
        self.volume_indices = volume_indices
        self.slice_enabled = slice_enabled
        self.swapaxis = swapaxis
        self.rnd_indices = rnd_indices
        self.keep_dims = keep_dims


    def __call__(self, representation : torch.Tensor, outer_iteration : int, inner_iteration : int) -> List[torch.Tensor]:

        slice_budget = self.slice_budget

        rep = math.ceil( float(slice_budget) / self.slab_thickness)

        ret = []
        ret_slice_inds = []
        for volume_index, slice_enabled, swapping, keep_dim in zip(self.volume_indices, self.slice_enabled, self.swapaxis, self.keep_dims):
            if slice_enabled:
                slice_dim = representation.shape[volume_index] # not sure if this is correct

                stride = self.stride
                index_mask = ((stride+1)*(torch.arange(self.slab_thickness, device=representation.device) - self.slab_thickness // 2)).repeat(rep)
                offset = (stride+1)*(self.slab_thickness//2)

                slice_inds = torch.randint(offset, slice_dim-offset, (rep,), device=representation.device).repeat_interleave(self.slab_thickness) + index_mask
                slice_inds = slice_inds[:slice_budget] # cut off if too many

                slices = representation.index_select(volume_index, slice_inds.int())

                ret_slice_inds.append(slice_inds)
                
                if not keep_dim:
                    slices = slices.moveaxis(volume_index, 0)

                elif not keep_dim:
                     slices = slices.squeeze(1) # not necessaryanymore

                if swapping:
                    slices = slices.swapaxes(-2,-3)

                ret.append(slices)

        return ret, ret_slice_inds

def get_slice_method(slice_cfg : Dict):
    if slice_cfg is not None:
        return RandomSlabsMethod(**slice_cfg.params)
    else:
        return None