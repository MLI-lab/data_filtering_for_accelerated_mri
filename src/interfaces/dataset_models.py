import pydantic
from typing import List, Dict, Optional, Tuple

## ----------------- Dataset config models ---------------------------- ##

class DatasetSliceModel(pydantic.BaseModel):
    freq: int

class DatasetVolumeModel(pydantic.BaseModel):
    path: str
    dataset_name : str
    split : str
    slices : Optional[Dict[str, DatasetSliceModel]] = None # in 3D all slices are taken

class DatasetModel(pydantic.BaseModel):
    """
        Attributes of the dataset files, encoded as pydantic model.

        Some comments:
        
        num_slices: kind of redundant, since it can be derived from the number of slices in the volume files
        
        readout_dim_fftshift_cor:
            - 3D a fft1 is applied to the slice-dim (which is assumed to be the Z-direction).
            - some datasets require to omit one of the fft shifts corresponding to the fft1 (e.g. Stanford datasets)
            - in 2D one could also correct this readout dim fftshift, but this is not implemented yet.
    """
    dataset_name : str
    uuid : str
    num_slices : Optional[int] = None                           #
    dataset_is_3d : bool = False                                # 
    readout_dim_fftshift_cor : Tuple[bool, bool] = [True, True] #
    files : Dict[str, DatasetVolumeModel]                       # key is the volume name