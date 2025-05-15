"""
Provides data and experimental utilities.
"""
from typing import Optional, Any

from src.train_eval_setups.diff_model_recon.physics_trafos.trafo_fwd.mri_2d_trafo import SubsampledFourierTrafo
from src.train_eval_setups.diff_model_recon.physics_trafos.trafos_prior_target.CroppedMagnitudeImageTrafo import CroppedMagnitudeImageTrafo
from src.train_eval_setups.diff_model_recon.physics_trafos.trafos_prior_target.IdentityTrafo import IdentityTrafo

try:
    import hydra.utils
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

from torch.utils.data import Dataset, TensorDataset
from omegaconf import DictConfig
from .trafo_fwd.base_trafo_fwd import BaseRayTrafo

def get_standard_target_trafo(cfg: DictConfig) -> BaseRayTrafo:
    if cfg.trafo_target.name in (CroppedMagnitudeImageTrafo.__name__):
        target_trafo = CroppedMagnitudeImageTrafo(
            **cfg.trafo_target.params
        )
        return target_trafo
    elif cfg.trafo_target.name in (IdentityTrafo.__name__):
        return IdentityTrafo()
    else:
        raise NotImplementedError

def get_standard_prior_trafo(cfg: DictConfig) -> BaseRayTrafo:
    if cfg.trafo_prior.name in (CroppedMagnitudeImageTrafo.__name__):
        prior_trafo = CroppedMagnitudeImageTrafo(
            **cfg.trafo_prior.params
        )
        return prior_trafo
    elif cfg.trafo_prior.name in (IdentityTrafo.__name__):
        return IdentityTrafo()
    else:
        raise NotImplementedError

def get_standard_trafo_fwd(cfg) -> BaseRayTrafo:
    if cfg.trafo_fwd.name in ('mri2d'):
        from src.train_eval_setups.diff_model_recon.physics_trafos.trafo_fwd.mri_2d_trafo import SubsampledFourierTrafo
        trafo_fwd = SubsampledFourierTrafo(**cfg.trafo_fwd.params)

    elif cfg.trafo_fwd.name in ('mri3d'):
        from src.train_eval_setups.diff_model_recon.physics_trafos.trafo_fwd.mri_3d_trafo import SubsampledFourierTrafo3D
        trafo_fwd = SubsampledFourierTrafo3D(**cfg.trafo_fwd.params)
    else:
        raise ValueError(f"Trafo {cfg.trafo_fwd.name} not implemented")

    return trafo_fwd

def get_standard_dataset_transform(
        cfg: DictConfig, 
        device: Optional[Any] = None,
        provide_pseudinverse : bool = False,
        provide_measurement : bool = False) -> Dataset:

    """
        By default the dataset shall return tuples (observation, ground_truth).
        On provide_pseudo_inverse=True, the dataset shall return (observation, ground_truth, pseudo_inverse)
    """
    if cfg.trafo_dataset.name == "mri2d":
       from .trafo_datasets.fastmri_2d_trafo import FastMRI2DDataTransform
       return FastMRI2DDataTransform(**cfg.trafo_dataset.params, provide_pseudinverse=provide_pseudinverse, provide_measurement=provide_measurement, device=device)
        
    elif cfg.trafo_dataset.name == "mri3d":
       from .trafo_datasets.fastmri_3d_trafo import FastMRI3DDataTransform
       return FastMRI3DDataTransform(**cfg.trafo_dataset.params, provide_pseudinverse=provide_pseudinverse, provide_measurement=provide_measurement, device=device)
    else: 
        raise NotImplementedError(f"Dataset {cfg.trafo_dataset.name} not implemented")
