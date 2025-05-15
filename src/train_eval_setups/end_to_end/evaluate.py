
import argparse
import torch
from .utils import get_dataloader
from tqdm.autonotebook import tqdm
from fastmri.evaluate import ssim, psnr
from runstats import Statistics
import pandas as pd
from torch.nn.functional import interpolate
from fastmri.data.transforms import center_crop
from .utils import setup
from pathlib import Path
import os
import lpips
from DISTS_pytorch import DISTS
import json
import uuid
from datetime import date
import yaml
import numpy as np
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
        self._lpips = _lpips
        self._dists = _dists
        
    def push(self, output, target, maxval, mask=None, scale_shift=None, pixel_based=True, feature_based=True):
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
        score = {}
        if mask is not None:
            output = output * mask
            target = target * mask
            
        if feature_based:
            with torch.no_grad():
                lpips_score = self.lpips(output, target).item()
                dists_score = self.dists(output, target).item()
            score['LPIPS'] = lpips_score
            score['DISTS'] = dists_score

        if pixel_based:
            output = output.squeeze(0).cpu().numpy()
            target = target.squeeze(0).cpu().numpy()
            ssim_score = ssim(target, output, maxval).item()
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


def classic_eval(dataset, model, forward_fn, data_transform, num_workers=4):
    model.eval()
    device = next(model.parameters()).device
    dataloader = get_dataloader(dataset, data_transform, batch_size=1, shuffle=False, num_workers=num_workers, load_sensitivity_maps=True)
    eval = Evaluation(device)
    for sample in tqdm(dataloader):
        target, maxval = sample.target, sample.max_value.item()
        target = target.unsqueeze(-3).to(device)
        with torch.no_grad():
            output = forward_fn(model, sample, device)
        output = center_crop(output, target.shape[-2:])

        dataset_name = dataset.split('/')[-1].split('.json')[0]
        filename = sample.fname[0].split('.h5')[0]
        slice_num = sample.slice_num[0].item()
        save_file = f'dataset={dataset_name}_fname={filename}_slice={slice_num}.pt'
        torch.save(output.cpu(), save_file)

        output_mask = sample.output_mask.to(output.device)
        output_mask = center_crop(output_mask, target.shape[-2:])
        eval.push(output, target, maxval, mask = output_mask)

    return eval


def pathology_eval(dataset, model, forward_fn, data_transform, num_workers=4):
    model.eval()
    device = next(model.parameters()).device
    dataloader = get_dataloader(dataset, data_transform, batch_size=1, shuffle=False, num_workers=num_workers, load_sensitivity_maps=True)
    with open(dataset) as f:
        json_data = json.load(f)
    eval = Evaluation(device)
    with tqdm(total=len(dataloader)) as pbar:
        for sample in dataloader:
            target, maxval, fname, slice_num = sample.target, sample.max_value.item(), sample.fname[0], sample.slice_num.item()
            target = target.unsqueeze(-3).to(device)
            with torch.no_grad():
                output = forward_fn(model, sample, device)
                
            output = center_crop(output, target.shape[-2:])
            dataset_name = dataset.split('/')[-1].split('.json')[0]
            filename = sample.fname[0].split('.h5')[0]
            slice_num = sample.slice_num[0].item()
            save_file = f'dataset={dataset_name}_fname={filename}_slice={slice_num}.pt'
    
            torch.save(output.cpu(), save_file)

            output_mask = sample.output_mask.to(output.device)
            output_mask = center_crop(output_mask, target.shape[-2:])
            
            # Compute LPIPS and DISTS on entire image
            eval.push(output, target, maxval, mask=output_mask, feature_based=True, pixel_based=False)

            # Get normalization parameters based on entire image
            scale, shift = eval._get_normalization_params(output, target, mask=output_mask)

            # Compute SSIM, PSNR on region with pathology; one slice can have multiple pathologies
            cases = json_data['files'][fname]['slices'][str(slice_num)]['pathologies']
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

                eval.push(output_patch, target_patch, maxval, scale_shift=(scale, shift), feature_based=False, pixel_based=True)

                pbar.update(1)

    return eval

def evaluate(
    path_to_checkpoint,
    setup_config,
    eval_config,
    eval_dataset_base_path,
    num_workers = 4,
    #accl_factors = [4],
    ):

    accl_factors = setup_config["accel_factors"] # TODO: this we should differently probably

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data_transform, forward_fn = setup('test', setup_config, accl_factors=accl_factors)
    model = model.to(device)
    cp = torch.load(path_to_checkpoint, map_location=device)
    model.load_state_dict(cp['model_state_dict'])

    global _lpips 
    _lpips = lpips.LPIPS(net='vgg').to(device)
    global _dists 
    _dists = DISTS().to(device)

    results = {
        'classic': {},
        'pathology': {}
    }

    def update_result(results, eval, task, dataset):
        results[task][Path(dataset).stem] = {}
        for metric, stats in eval.metric.items():
            results[task][Path(dataset).stem][metric] = stats.mean()

    # Classic evaluation
    path_to_datasets = eval_config.classic
    for dataset in path_to_datasets:
        dataset_path = os.path.join(eval_dataset_base_path, dataset)
        eval = classic_eval(dataset_path, model, forward_fn, data_transform, num_workers=num_workers)
        update_result(results, eval, 'classic', dataset)
        print(eval.metric['SSIM_normal'].mean())

    # Pathology evaluation
    path_to_datasets = eval_config.pathology
    for dataset in path_to_datasets:
        dataset_path = os.path.join(eval_dataset_base_path, dataset)
        eval = pathology_eval(dataset_path, model, forward_fn, data_transform, num_workers=num_workers)
        update_result(results, eval, 'pathology', dataset)
        print(eval.metric['SSIM_normal'].mean())

    return results

def main(args, setup_config, setup_config_file, eval_config, eval_dataset_base_path):
    
    # outfile is simply "eval_results.json", since we are already in the current directory
    outfile = args.outfile
    path_to_checkpoint = args.model_path
    num_workers = args.num_workers
    print(path_to_checkpoint)
    results_json = {
        "name": Path(outfile).stem, # overwritten by base setup for final summary
        "model": Path(setup_config_file).stem, # overwritten by base setup for final summary
        "uuid": str(uuid.uuid4()),
        "creation_date": str(date.today()),
    }

    results = evaluate(path_to_checkpoint, setup_config, eval_config, num_workers=num_workers, eval_dataset_base_path=eval_dataset_base_path)
    results_json['eval_metrics'] = results
    results_json = json.dumps(results_json, indent=4)

    with open(outfile, 'w') as f:
        f.write(results_json)

    return results_json
