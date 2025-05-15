import argparse
import torch
from .utils.utils import get_dataloader

from ..common_utils.datasets.dataset_factory import get_dataset
from tqdm.autonotebook import tqdm
from fastmri.evaluate import ssim, psnr
from runstats import Statistics
import pandas as pd
from torch.nn.functional import interpolate
from fastmri.data.transforms import center_crop
from pathlib import Path
import os
import lpips
from DISTS_pytorch import DISTS
import json
import uuid
from datetime import date
import yaml
import numpy as np
import wandb

from omegaconf import DictConfig, OmegaConf
from .utils.wandb_utils import wandb_kwargs_via_cfg
from src.train_eval_setups.diff_model_recon.reconstruct_reg_tuning import val_reg_tuning
import logging

class Evaluation:
    def __init__(self, device):
        self.metric = {
            'SSIM': Statistics(),
            'SSIM_normal': Statistics(),
            'PSNR': Statistics(),
            'PSNR_normal': Statistics(),
            'LPIPS': Statistics(),
            'LPIPS_normal': Statistics(),
            'DISTS': Statistics(),
            'DISTS_normal': Statistics(),
        }
        self._lpips = lpips.LPIPS(net='vgg').to(device)
        self._dists = DISTS().to(device)
        
    def push(self, output, target, maxval, mask=None, scale_shift=None, pixel_based=True, feature_based=True):
        # before evaluating, we reshape the 3D and 2D volumes to an image-like format
        assert output.shape[0] == 1 and target.shape[0] == 1, "Batch size must be 1 for 3D case."
        if output.ndim == 4: # 3D case
            # assumed shape (1, Z, Y, X)
            output = output[0, :, None, :, :] # (Z, 1, Y, X)
            target = target[0, :, None, :, :] # (Z, 1, Y, X)
        elif output.ndim == 3: # 2D case
            # assumed shape (1, Y, X)
            output = output[None] # (1, 1, Y, X)
            target = target[None] # (1, 1, Y, X)

        # evaluation
        score = self._eval(output, target, maxval, mask=mask, pixel_based=pixel_based, feature_based=feature_based)
        if pixel_based:
            self.metric['SSIM'].push(score['SSIM'])
            self.metric['PSNR'].push(score['PSNR'])
        if feature_based:
            self.metric['LPIPS'].push(score['LPIPS'])
            self.metric['DISTS'].push(score['DISTS'])
        
        score = self._normalized_eval(output, target, maxval, mask=mask, scale_shift=scale_shift, pixel_based=pixel_based, feature_based=feature_based)
        if pixel_based:
            self.metric['SSIM_normal'].push(score['SSIM'])
            self.metric['PSNR_normal'].push(score['PSNR'])
        if feature_based:
            self.metric['LPIPS_normal'].push(score['LPIPS'])
            self.metric['DISTS_normal'].push(score['DISTS'])

    def _eval(self, output, target, maxval, mask=None, pixel_based=True, feature_based=True):
        # we can assume the shape (B, 1, Y, X) for 2D and 3D
        score = {}
        if mask is not None:
            output = output * mask
            target = target * mask
        if feature_based:
            with torch.no_grad():
                lpips_score = self.lpips(output, target).mean().item() # take the mean along the Z-dir
                dists_score = self.dists(output, target).mean().item() # take the mean along the Z-dir
            score['LPIPS'] = lpips_score
            score['DISTS'] = dists_score

        if pixel_based:
            output = output.squeeze(-3).cpu().numpy()
            target = target.squeeze(-3).cpu().numpy()
            ssim_score = ssim(target, output, maxval).item() # ssim expects (B, Y, X)
            psnr_score = psnr(target, output, maxval).item()
            score['SSIM'] = ssim_score
            score['PSNR'] = psnr_score
    
        return score
    
    def lpips(self, x, y):
        x = self._scale(x).repeat(1,3,1,1) * 2 - 1
        y = self._scale(y).repeat(1,3,1,1) * 2 - 1
        
        return self._lpips(x,y)
    
    def dists(self, x, y):
        x = self._scale(x).repeat(1,3,1,1)
        y = self._scale(y).repeat(1,3,1,1)
        
        return self._dists(x,y)

    def _normalized_eval(self, output, target, maxval, mask=None, scale_shift=None, pixel_based=True, feature_based=True):
        if scale_shift is None:
            scale, shift = self._get_normalization_params(output, target, mask=mask)
        else:
            scale, shift = scale_shift
            
        output = self._normalize(output, scale, shift, mask=mask)
        score = self._eval(output, target, maxval, mask=mask, pixel_based=pixel_based, feature_based=feature_based)
        return score

    def _get_normalization_params(self, output, target, mask=None):
        if mask is not None: # not usable for multibatch eval
            mt = target[mask].mean()[None,None,None,None]
            st = target[mask].std()[None,None,None,None]
            mo = output[mask].mean()[None,None,None,None]
            so = output[mask].std()[None,None,None,None]
        else:
            mt = target.mean(axis=(-2, -1), keepdims=True)
            st = target.std(axis=(-2, -1), keepdims=True)
            mo = output.mean(axis=(-2, -1), keepdims=True)
            so = output.std(axis=(-2, -1), keepdims=True)
            
        scale = st/so
        shift = mt - mo * st/so
        return scale, shift

    def _normalize(self, x, scale, shift, mask=None):
        if mask is not None:
            x[mask] = x[mask] * scale + shift
            return x
        else:
            return x * scale + shift
    
    def _scale(self, image):
        image_max = 1
        scale = image_max / torch.quantile(image, 0.999).item()
        image = torch.clip((scale * image), 0, image_max)

        return image


def classic_eval(dataset_path, forward_fn, dataset_trafo, num_workers=4, device="cuda",cfg=None):
    #model.eval()
    dataset = get_dataset(dataset_path, transform=dataset_trafo)
    dataloader = get_dataloader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    eval = Evaluation(device)
    sample_nr = 0
    for sample in tqdm(dataloader):
        # dataloader returns observation, target, pseudorec, attrs
        #target, maxval = sample.target, sample.max_value.item()
        # we store the old iteration count to reset it after the validation (during validation it may be smaller)
        old_iteration_count = cfg['reconstruct']['variational_optim']['iterations']
        #val_reg_strength = val_reg_tuning_cv_bisection(sample, forward_fn, cfg, device)
        val_reg_strength = val_reg_tuning(sample, forward_fn, cfg, device)

        logging.info(f"Validation regularization strength for sample_nr {sample_nr}: {val_reg_strength}")
        sample_nr += 1
        
        _, target, _, attrs = sample
        maxval = attrs["max"].item()
        # target = target.unsqueeze(-3).to(device)
        cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = val_reg_strength
        cfg['reconstruct']['variational_optim']['iterations'] = old_iteration_count
        output = forward_fn(sample, cfg)
        output = center_crop(output[0], target.shape[-2:])
        dataset_name = dataset_path.split('/')[-1].split('.json')[0]
        filename = attrs['fname'][0].split('.h5')[0]
        slice_num = attrs['slice_num'].item()
        save_file = f'dataset={dataset_name}_fname={filename}_slice={slice_num}.pt'
        torch.save(output[None].cpu(), save_file)

        output_mask = attrs['output_mask'].to(output.device)
        output_mask = center_crop(output_mask, target.shape[-2:])

        eval.push(output, target, maxval, mask = output_mask)

        # break # TODO: Added for debugging!!!

    return eval


def pathology_eval(dataset_path, forward_fn, dataset_trafo, num_workers=4, device="cuda",cfg=None):
    #model.eval()
    #device = next(model.parameters()).device
    dataset = get_dataset(dataset_path, transform=dataset_trafo)
    dataloader = get_dataloader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    eval = Evaluation(device)
    sample_nr = 0
    with open(dataset_path) as f:
        json_data = json.load(f)
    with tqdm(total=len(dataloader)) as pbar:
        for sample in dataloader:
            old_iteration_count = cfg['reconstruct']['variational_optim']['iterations']
            val_reg_strength = val_reg_tuning(sample, forward_fn, cfg, device)
    
            logging.info(f"Validation regularization strength for sample_nr {sample_nr}: {val_reg_strength}")
            sample_nr += 1
            
            _, target, _, attrs = sample
            cfg['reconstruct']['diffusion_reg_params']['reg_strength'] = val_reg_strength
            cfg['reconstruct']['variational_optim']['iterations'] = old_iteration_count
            output = forward_fn(sample,cfg)
            
            output = center_crop(output[0], target.shape[-2:])
                
            maxval = attrs["max"].item()
            fname = attrs['fname'][0]
            slice_num = attrs['slice_num'].item()

            dataset_name = dataset_path.split('/')[-1].split('.json')[0]
            filename = fname.split('.h5')[0]
            save_file = f'dataset={dataset_name}_fname={filename}_slice={slice_num}.pt'
            torch.save(output[None].cpu(), save_file)

            output_mask = attrs['output_mask'].to(output.device)
            output_mask = center_crop(output_mask, target.shape[-2:])
            
            # Compute LPIPS and DISTS on entire image
            eval.push(output, target, maxval, mask=output_mask, feature_based=True, pixel_based=False)


            # Compute SSIM, PSNR on region with pathology; one slice can have multiple pathologies
            cases = json_data['files'][fname]['slices'][str(slice_num)]['pathologies']
            assert len(output.shape) == 3, f'output in pathology evaluation has to be (1, y, x), current shape is {output.shape}'
            assert len(target.shape) == 3, f'target in pathology evaluation has to be (1, y, x), current shape is {output.shape}'
            output = output.unsqueeze(1)
            target = target.unsqueeze(1)

            # Get normalization parameters based on entire image
            scale, shift = eval._get_normalization_params(output, target, mask=output_mask)

            for case in cases:
                x, y, w, h = case['loc']
                target_patch = torch.flip(target, dims=(-2,))
                target_patch = target_patch[..., y:y+h, x:x+w]                
                output_patch = torch.flip(output, dims=(-2,))
                output_patch = output_patch[..., y:y+h, x:x+w]
                
                # Image patch must larger than SSIM window-size 
                axis_min = min(target_patch.shape[-2:])
                if axis_min < 7:
                    scale_factor = 7/axis_min
                    target_patch = interpolate(target_patch, scale_factor=scale_factor, mode='nearest-exact')
                    output_patch = interpolate(output_patch, scale_factor=scale_factor, mode='nearest-exact')

                output_patch = output_patch.squeeze(1)
                target_patch = target_patch.squeeze(1)

                eval.push(output_patch, target_patch, maxval, scale_shift=(scale, shift), feature_based=False, pixel_based=True)

                pbar.update(1)

    return eval

# results = evaluate(path_to_checkpoint, eval_dataset_config, eval_dataset_config_paths_base, num_workers=num_workers)

def evaluate(
    cfg,
    eval_dataset_config,
    eval_dataset_config_paths_base,
    num_workers
    ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from .reconstruct.reconstruct import reconstruction_setup
    forward_fn, dataset_trafo = reconstruction_setup(cfg, device=device)

    results = {
        'classic': {},
        'pathology': {}
    }

    def update_result(results, eval, task, dataset):
        results[task][Path(dataset).stem] = {}
        for metric, stats in eval.metric.items():
            results[task][Path(dataset).stem][metric] = stats.mean()

    # Classic evaluation
    path_to_datasets = eval_dataset_config.classic
    for dataset in path_to_datasets:
        dataset_path = os.path.join(eval_dataset_config_paths_base, dataset)
        eval = classic_eval(dataset_path, forward_fn, dataset_trafo, num_workers=num_workers, device=device, cfg=cfg)
        update_result(results, eval, 'classic', dataset)

    # Pathology evaluation
    path_to_datasets = eval_dataset_config.pathology
    for dataset in path_to_datasets:
        dataset_path = os.path.join(eval_dataset_config_paths_base, dataset)
        eval = pathology_eval(dataset_path, forward_fn, dataset_trafo, num_workers=num_workers, device=device, cfg=cfg)
        update_result(results, eval, 'pathology', dataset)

    return results

def main(cfg, eval_dataset_config, eval_dataset_config_paths_base):
    
    # loading data is not an issue so we fix a num_workers of 0 here.
    # TODO: make that part of the hydra cfg. later
    num_workers = 0
    outfile = 'eval_results.json'

    results_json = {
        "name": "", # will be set by base setup
        "model": "", # is set by the base setup
        "uuid": str(uuid.uuid4()),
        "creation_date": str(date.today()),
    }

    OmegaConf.resolve(cfg)

    # store yaml config in current directory
    with open('hydra_config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    wandb_kwargs = wandb_kwargs_via_cfg(cfg)

    with wandb.init(**wandb_kwargs) as run:
        results = evaluate(cfg, eval_dataset_config, eval_dataset_config_paths_base, num_workers=num_workers)
        results_json['eval_metrics'] = results
        results_json = json.dumps(results_json, indent=4)

        with open(outfile, 'w') as f:
            f.write(results_json)

        return results_json