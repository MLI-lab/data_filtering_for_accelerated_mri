import wandb
from torch import Tensor
import torch
from src.train_eval_setups.diff_model_recon.utils.utils import normalize_clamp, midslice2selct
from typing import Dict, Optional, Tuple
import k3d
import numpy as np
import logging

from omegaconf import DictConfig, OmegaConf

def tensor3d_to_wandb_video(x : Tensor, fps : Optional[int] = None, duration : Optional[int] = 5) -> wandb.Video:

    #if fps is None:

    if fps is None:
        fps = int(x.shape[0] // duration)

    # assume shape (Z, X, Y, C)
    if x.ndim == 3:
        x = x.unsqueeze(-1)
    assert x.ndim == 4

    x = x.moveaxis(-1, 1)
    if x.shape[1] == 1: 
        x = x.repeat_interleave(3, dim=1)

    if x.max() != 0.0:
        x = (x / x.max() * 255).byte()
    else:
        x = x.byte()
    return wandb.Video(x.cpu().numpy(), fps=fps, format="webm")

def add_tensor3d_wandb_videos_to_dict(basekey: str, x : Tensor, ret_dict : Dict, fps : Optional[int] = None, duration : Optional[float] = 5, video_z : bool = True, video_y : bool = True, video_x : bool = True, take_meanslices : bool = False, take_videos : bool = False):
    # assume data is (Z, Y, X) = (210, 640, 368)
    #pass
    if take_meanslices:
        Z, Y, X = x.shape
        if video_z:
            ret_dict[basekey + f"_median_coronal_zyx"] = wandb.Image(normalize_clamp(x[Z//2,...]).cpu().numpy())
        if video_y:
            ret_dict[basekey + f"_median_sagittal_xyz"] = wandb.Image(normalize_clamp(x.permute(2,1,0)[X//2]).cpu().numpy())
        if video_x:
            ret_dict[basekey + f"_median_axial_yzx"] = wandb.Image(normalize_clamp(x.permute(1,0,2)[Y//2]).cpu().numpy())
            
    if take_videos:
        if video_z:
            ret_dict[basekey + f"_coronal_zyx"] = tensor3d_to_wandb_video(x, fps=fps, duration=duration)
        if video_y:
            ret_dict[basekey + f"_sagittal_xyz"] = tensor3d_to_wandb_video(x.permute(2, 1, 0), fps=fps, duration=duration)
        if video_x:
            ret_dict[basekey + f"_axial_yzx"] = tensor3d_to_wandb_video(x.permute(1, 0, 2), fps=fps, duration=duration)

def tensor_to_wandbimages_dict(basekey: str, x : Tensor, suffix_mag : str = "mag", suffix_phase : str = "phase", video_fps : Optional[int] = None, video_duration : Optional[float] = 5, video_z : bool = True, video_x : bool = True, video_y : bool = True, take_meanslices : bool = False, take_videos : bool = False, show_phase : bool = False) -> Dict:

    ret_dict = {}
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    if x.ndim == 3 or (x.ndim==4 and x.shape[0] == 1):
        if x.ndim==4:
            x = x.squeeze(0)
        # could either be (B, H, W) or (H, W, C) or (Z, X, Y)
        if x.shape[-1] == 1 or x.shape[-1] == 3:
            # real grayscale or rgb image
            ret_dict[basekey] = wandb.Image(normalize_clamp(x).cpu().numpy())
        elif x.shape[-1] == 2:
            ret_dict[basekey + f"_{suffix_mag}"] = wandb.Image(normalize_clamp(x.norm(dim=-1)).cpu().numpy())
            if show_phase:
                angles = torch.view_as_complex(x.reshape(-1, 2).contiguous()).angle().view(x.shape[:-1])
                ret_dict[basekey + f"_{suffix_phase}"] = wandb.Image(normalize_clamp(angles).cpu().numpy())
        else:
            #ret_dict[basekey] = tensor3d_to_wandb_video(x, fps=video_fps, duration=video_duration)
            add_tensor3d_wandb_videos_to_dict(basekey, x, ret_dict, fps=video_fps, duration=video_duration, video_z=video_z, video_x=video_x, video_y=video_y, take_meanslices=take_meanslices, take_videos=take_videos)
        return ret_dict
    #elif x.ndim == 4 and x.shape[-1] <= 3: # maybe not the best way to distinguish
        ## shape (B, H, W, C) or (Z, X, Y, C)
        ## not optimal..
        #batch_size = x.shape[0]
        #for i in range(batch_size):
            #x_i = x[i, ...]
            #if x_i.shape[-1] == 1 or x_i.shape[-1] == 3:
                ## real grayscale or rgb image
                #ret_dict[basekey + f"_{i}"] = wandb.Image(normalize_clamp(x).cpu().numpy())
            #elif x_i.shape[-1] == 2:
                #ret_dict[basekey + f"_{i}_{suffix_mag}"] = wandb.Image(normalize_clamp(x_i.norm(dim=-1)).cpu().numpy())
                #if show_phase:
                    #angles = torch.view_as_complex(x_i.reshape(-1, 2).contiguous()).angle().view(x_i.shape[:-1])
                    #ret_dict[basekey + f"_{i}_{suffix_phase}"] = wandb.Image(normalize_clamp(angles).cpu().numpy())
            #else:
                #raise NotImplementedError()
        #return ret_dict
    elif x.ndim == 5 or x.ndim == 4:
        # (B, Z, X, Y, C) or (Z, X, Y, C)

        if x.ndim == 4:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        for i in range(batch_size):
            batch_prefix = f"_{i}" if batch_size > 1 else ""
            x_i = x[i, ...] # (Z, X, Y, C)
            if x_i.shape[-1] == 1 or x_i.shape[-1] == 3:
                #ret_dict[basekey + batch_prefix] = tensor3d_to_wandb_video(x_i, fps=video_fps, duration=video_duration)
                add_tensor3d_wandb_videos_to_dict(basekey + batch_prefix, x_i, ret_dict, fps=video_fps, duration=video_duration, video_z=video_z, video_x=video_x, video_y=video_y, take_meanslices=take_meanslices, take_videos=take_videos)
            elif x_i.shape[-1] == 2:
                #ret_dict[basekey + f"{batch_prefix}_{suffix_mag}"] = tensor3d_to_wandb_video(x_i.norm(dim=-1), fps=video_fps, duration=video_duration)
                add_tensor3d_wandb_videos_to_dict(basekey + batch_prefix + f"_{suffix_mag}", x_i.norm(dim=-1), ret_dict, fps=video_fps, duration=video_duration, video_z=video_z, video_x=video_x, video_y=video_y, take_meanslices=take_meanslices, take_videos=take_videos)
                if show_phase:
                    angles = torch.view_as_complex(x_i.reshape(-1, 2).contiguous()).angle().view(x_i.shape[:-1])
                    #ret_dict[basekey + f"{batch_prefix}_{suffix_phase}"] = tensor3d_to_wandb_video(angles, fps=video_fps, duration=video_duration)
                    add_tensor3d_wandb_videos_to_dict(basekey + batch_prefix + f"_{suffix_phase}", angles, ret_dict, fps=video_fps, duration=video_duration, video_z=video_z, video_x=video_x, video_y=video_y, take_meanslices=take_meanslices, take_videos=take_videos)
            else:
                raise NotImplementedError()
        return ret_dict
    elif x.ndim == 6:
        (B, Z, C, X, Y, C) = x.shape

        # multiple images like batch or batch of videos
        #raise NotImplementedError("tensor_to_wandbimages_dict not implemented for 5 dimensions")
    elif x.ndim == 2:
        return {basekey : wandb.Image(normalize_clamp(x).cpu().numpy())}
    else:
        raise NotImplementedError("tensor_to_wandbimages only implemented for 2, 3 or 4 dimensions")

def tensor_to_wandbimage(x : Tensor):
    # assume shape (1, H, W, C)
    #x = midslice2selct(x)[0, ...]
    if x.ndim == 3 and (x.shape[0] != 1 and x.shape[0] != 3):
        # select midslice
        x = x[x.shape[0] // 2, ...]
    if x.shape[-1] == 2:
        # assume complex
        x = x.norm(dim=-1)
    return wandb.Image(normalize_clamp(x).cpu().numpy())

#wandb.Image(
    #torchvision.utils.make_grid(
        #x_mean.norm(dim=-3).squeeze(), normalize=True, scale_each=True
    #)
    #.permute(1, 2, 0)
    #.cpu()
    #.numpy()
#),

def flatten_hydra_config(cfg : DictConfig) -> dict:
    out_dict = {}
    _flatten_hydra_config(cfg, out_dict)
    return out_dict

def _flatten_hydra_config(cfg : DictConfig, out_dict : dict, prefix : str = ""):
    new_prefix = prefix + "." if len(prefix) > 0 else ""
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            _flatten_hydra_config(value, out_dict, new_prefix + key)
        else:
            out_dict[new_prefix + key] = value


def wandb_kwargs_via_cfg(cfg : DictConfig, use_group_name: bool = True) -> dict:
    """
        Sets the wandb arguments based on the configuration.
        
    """
    if cfg.wandb.group_name is None or not use_group_name:
        wandb_group = None
        wandb_name = None
    else:
        # group_name is set via experiment, e.g. exp1/case1/config.yaml
        # this generates group=exp1/case1 and name=config.yaml
        wandb_group, wandb_name = cfg.wandb.group_name.rsplit("/",1)

    wandb_name_full = wandb_name if cfg.wandb.descr_short is None else f"{wandb_name}_{cfg.wandb.descr_short}"

    wandb_kwargs = {
        'project': cfg.wandb.project,
        'entity': cfg.wandb.entity,
        'name': wandb_name_full, # cfg.wandb.name if cfg.wandb.name else None,
        'mode': 'online' if cfg.wandb.log else 'disabled',
        'settings': wandb.Settings(code_dir=cfg.wandb.code_dir),
        'group' : wandb_group,
        'config' : flatten_hydra_config(cfg)
    }
    return wandb_kwargs

def volume_to_k3d_html(volume : np.array, filter_upper_percentile : float = 95.0, filter_lower_percentile : float = 5.0, bounds_shape : Optional[Tuple[int, int, int]] = None, filename = "k3d.html"):
    plot = k3d.plot(lighting=0.0)
    # the next command can take some time
    logging.info("Creating k3d volume...")
    if bounds_shape is None:
        bounds = None
    else:
        bounds = [0, bounds_shape[2], 0, bounds_shape[1], 0, bounds_shape[0]]
    plt_volume = k3d.volume(volume,
            color_range=[
                np.percentile(volume, filter_lower_percentile),
                np.percentile(volume, filter_upper_percentile)
            ],
            interpolation=False,
            compression_level=9,
            alpha_coef=100,
            gradient_step=0.00005,
            bounds = bounds, #[0, volume.shape[2], 0, volume.shape[1], 0, volume.shape[0]],
            #color_map=k3d.paraview_color_maps.X_Ray)
            color_map=k3d.paraview_color_maps.Grayscale)
    plot += plt_volume
    plot.display()
    logging.info("Finished creating k3d volume")
    return plot.get_snapshot()

def volume_to_wandb_as_k3d_html(volume : np.array, filter_upper_percentile : float = 99.9, filter_lower_percentile : float = 0.1, filename = "k3d.html", bounds_shape = None):
    with open(filename, 'w') as f:
        f.write(volume_to_k3d_html(volume, filter_upper_percentile, filter_lower_percentile, bounds_shape=bounds_shape))
    return wandb.Html(open(filename), inject=False)

def log_volume_in_wandb_as_k3dhtml(volume : np.array, wandb_run, step: int, desc: str, filter_upper_percentile : float = 99.9, filter_lower_percentile : float = 0.1, bounds_shape = None, filename = "k3d.html"):
    wandb_run.log({
        desc : volume_to_wandb_as_k3d_html(volume, filter_upper_percentile, filter_lower_percentile, filename, bounds_shape=bounds_shape),
        "global_step" : step
    })

import multiprocessing as mp
pool = None

def volume_to_wandb_as_k3d_html__async(volume : np.array, wandb_run, step, desc, filter_upper_percentile : float = 99.9, filter_lower_percentile : float = 0.1, filename = "k3d.html"):

    global pool
    if pool is None:
        pool = mp.Pool(processes=1)

    volume_cp = np.copy(volume)
    pool.apply_async(log_volume_in_wandb_as_k3dhtml, (volume_cp, wandb_run, step, desc, filter_upper_percentile, filter_lower_percentile, filename))

    return pool