from glob import glob
import torch
import torch
from fastmri.evaluate import ssim, psnr
from runstats import Statistics
from fastmri.data.transforms import center_crop, to_tensor
from fastmri.data import transforms, subsample
from fastmri.data import transforms as T
from pathlib import Path
import os
from lpips import LPIPS
from DISTS_pytorch import DISTS
import json
import uuid
from datetime import date
import numpy as np
import sys
sys.path.insert(0,'../')
from src.train_eval_setups.end_to_end.utils import SliceDataset
import re
import argparse
import shutil
import pandas as pd
from tqdm.autonotebook import tqdm

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


#================================================================================
# Reading data
#================================================================================
def eval_to_df(path, task='classic', keys=None):
    fc = path
    with open(fc) as f:
        control = json.load(f)
        
    keys_c = list(control['eval_metrics']['classic'].keys())
    keys_p = list(control['eval_metrics']['pathology'].keys())
    keys_ood = [s for s in control['eval_metrics']['classic'].keys() if 'smurf' in s or 'stanford' in s or 'gre_rot' in s or 'prostate' in s or 'nyu' in s or 'ocmr' in s]
    keys_id = [s for s in control['eval_metrics']['classic'].keys() if s not in keys_ood]

    if keys is None:
        if task == 'classic':
            keys = keys_c
        elif task == 'pathology':
            keys = keys_p
        elif task == 'ood':
            keys = keys_ood
        elif task == 'id':
            keys = keys_id
    
    evals = {}
    for k in keys:
        eval = {}
        if task == 'pathology':
            for metric in control['eval_metrics']['pathology'][k]:
                eval[metric] = control['eval_metrics']['pathology'][k][metric]
        else:
            for metric in control['eval_metrics']['classic'][k]:
                eval[metric] = control['eval_metrics']['classic'][k][metric]
        evals[k] = eval
        
    return pd.DataFrame(evals).T

def get_scores(files, task, metric, keys=None):
    result = []
    for f in files:
        df = eval_to_df(f, task=task, keys=keys)
        result.append(df.describe()[metric]['mean'])
    return result


#================================================================================
# Bootstrap
#================================================================================
class Evaluation:
    def __init__(self, device):
        self.metric = {
            'SSIM': [],
            'SSIM_normal': [],
            'PSNR': [],
            'PSNR_normal': [],
            'LPIPS': [],
            'LPIPS_normal': [],
            'DISTS': [],
            'DISTS_normal': [],
        }
        self._lpips = lpips
        self._dists = dists
        
    def push(self, output, target, maxval, mask=None, scale_shift=None, pixel_based=True, feature_based=True):
        score = self._eval(output, target, maxval, mask=mask, pixel_based=pixel_based, feature_based=feature_based)
        if pixel_based:
            self.metric['SSIM'].append(score['SSIM'])
            self.metric['PSNR'].append(score['PSNR'])
        if feature_based:
            self.metric['LPIPS'].append(score['LPIPS'])
            self.metric['DISTS'].append(score['DISTS'])
        
        score = self._normalized_eval(output, target, maxval, mask=mask, scale_shift=scale_shift, pixel_based=pixel_based, feature_based=feature_based)
        if pixel_based:
            self.metric['SSIM_normal'].append(score['SSIM'])
            self.metric['PSNR_normal'].append(score['PSNR'])
        if feature_based:
            self.metric['LPIPS_normal'].append(score['LPIPS'])
            self.metric['DISTS_normal'].append(score['DISTS'])

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


class DataTransform(object):
    """
    Data Transformer that masks input k-space.
    """

    def __init__(self,):
        pass

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        """
        Data Transformer that simply returns the input masked k-space data and
        relevant attributes needed for running MRI reconstruction algorithms
        implemented in BART.

        Args:
            masked_kspace (numpy.array): Input k-space of shape (num_coils, rows,
                cols, 2) for multi-coil data or (rows, cols, 2) for single coil
                data.
            target (numpy.array, optional): Target image.
            attrs (dict): Acquisition related information stored in the HDF5
                object.
            fname (str): File name.
            slice_num (int): Serial number of the slice.

        Returns:
            tuple: tuple containing:
                masked_kspace (torch.Tensor): Sub-sampled k-space with the same
                    shape as kspace.
                reg_wt (float): Regularization parameter.
                fname (str): File name containing the current data item.
                slice_num (int): The index of the current slice in the volume.
                crop_size (tuple): Size of the image to crop to given ISMRMRD
                    header.
                num_low_freqs (int): Number of low-resolution lines acquired.
        """
        crop_size = target.shape
        max_value = attrs["max"]
        sens_maps = attrs['sensitivity_maps']
        return (sens_maps, target, max_value, fname, slice_num, crop_size)

def update_result(results, eval_1, eval_2, task, dataset):
    results[task][Path(dataset).stem] = {}
    for metric, stats in eval_1.metric.items():
        results[task][Path(dataset).stem][metric] = np.array(eval_2.metric[metric]) - np.array(stats)
        
def score(eval, load_file, target, max_value, sens_maps, crop_size, device):
        output = torch.load(load_file).to(device)
        target = to_tensor(np.abs(target))[None,None].to(device)
        target = center_crop(target, crop_size)
        maxval = max_value.item()
        output_mask = np.abs(np.sum(sens_maps * np.conj(sens_maps), axis=0, keepdims=True))
        output_mask = output_mask > 0.5
        output_mask = transforms.to_tensor(output_mask)[None].to(device)
        output_mask = center_crop(output_mask, crop_size)
        eval.push(output, target, maxval, mask = output_mask)
        

def get_difference(load_dir_1, load_dir_2, device):
    
    files = natural_sort(glob('../datasets/evals/classic/nuanced_new/*.json'))
    results = {
            'classic': {},
            'pathology': {}
    }
    for f in tqdm(files):
        eval_1 = Evaluation(device)
        eval_2 = Evaluation(device)
        data_transform = DataTransform()
        d = SliceDataset(f, transform=data_transform)
        for i in range(len(d)):
            sens_maps, target, max_value, fname, slice_num, crop_size = d[i]
            load_file_1 = os.path.join(load_dir_1, 'dataset=' + Path(f).stem + '_fname=' + Path(fname).stem + f'_slice={slice_num}' + '.pt')
            load_file_2 = os.path.join(load_dir_2, 'dataset=' + Path(f).stem + '_fname=' + Path(fname).stem + f'_slice={slice_num}' + '.pt')
            score(eval_1, load_file_1, target, max_value, sens_maps, crop_size, device)
            score(eval_2, load_file_2, target, max_value, sens_maps, crop_size, device)
        update_result(results, eval_1, eval_2, 'classic', f)
    return results


def bootstrap(results, num_bootstrap_samples=10000):
    results_per_sample = []
    metrics = list(list(results['classic'].values())[0].keys())
    for dataset_name, res in results['classic'].items():
        n_scores = len(res[metrics[0]])
        for i in range(n_scores):
            sample_data = [None]*(len(metrics)+1)
            for j, metric in enumerate(metrics):
                sample_data[j] = res[metric][i]
            sample_data[-1] = dataset_name
            results_per_sample.append(sample_data)
    results_per_sample = np.array(results_per_sample)
    
    # Start
    rng = np.random.default_rng(seed=42)
    n = len(results_per_sample)
    boots_means = {metric: [] for metric in metrics}
    
    for i in tqdm(range(num_bootstrap_samples)):
        sample_indices = rng.choice(n, n, replace=True)  # Resample with replacement
        boots = results_per_sample[sample_indices]
        boots_dict = {}
        for sample in boots:
            dataset_name = sample[-1]
            if dataset_name not in boots_dict:
                boots_dict[dataset_name] = []
            sample = np.array(sample[:-1]).astype(np.float64)
            boots_dict[dataset_name].append(sample)
        
        for k, v in boots_dict.items():
            boots_dict[k] = np.array(v).mean(axis=0)
        
        list_of_means = []
        for _, v in boots_dict.items():
            list_of_means.append(v)
        
        means = np.array(list_of_means).mean(axis=0)
        for i, metric in enumerate(metrics):
            boots_means[metric].append(means[i])  # Compute mean for each metric
    
    return pd.DataFrame(boots_means)

def bootstrap_eval(image_dir_control, image_dir_test, device='cuda'):
    global lpips, dists
    lpips = LPIPS(net='vgg').to(device)
    dists = DISTS().to(device)
    results = get_difference(image_dir_control, image_dir_test, device)
    df_bootstrap = bootstrap(results)
    
    return df_bootstrap




