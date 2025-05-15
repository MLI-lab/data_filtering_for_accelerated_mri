import sys
import os
#sys.path.insert(0,'/kang/data_filtering_mri/bart-0.6.00/python/')
#os.environ['TOOLBOX_PATH'] = "/kang/data_filtering_mri/bart-0.6.00"
#sys.path.insert(0,'/dss/dsshome1/06/ge83sel2/git/bart-0.6.00/python/')
#os.environ['TOOLBOX_PATH'] = "/dss/dsshome1/06/ge83sel2/git/bart-0.6.00/"
#sys.path.insert(0, os.path.join(os.getcwd(), "libs", "bart-0.6.00", "python"))
#os.environ['TOOLBOX_PATH'] = os.path.join(os.getcwd(), "libs", "bart-0.6.00")
from pathlib import Path
bart_dir = os.path.join(Path(os.path.realpath(__file__)).parents[4], "bart-0.6.00")
sys.path.insert(0, os.path.join(bart_dir, "python"))
os.environ['TOOLBOX_PATH'] = bart_dir
import bart
import numpy as np
import matplotlib.pyplot as plt
#import sigpy.mri as mr
import torch
import fastmri
from tqdm import tqdm
import multiprocessing as mp

# from https://github.com/MLI-lab/untrained-motion-correction/blob/master/only_trans_more_states.ipynb
def compute_sens_maps(masked_ksp):
    ### compute sensitivity maps
    masked_ksp = masked_ksp[...,0] + 1j*masked_ksp[...,1]
    sens_maps = bart.bart(1, f'ecalib -d0 -m1', np.array([np.moveaxis(masked_ksp.detach().cpu().numpy(),0,2)]))
    return np.moveaxis(sens_maps[0],2,0)

def compute_sens_maps_3d(masked_ksp):
    kspace_full_complex = torch.view_as_complex(masked_ksp)
    kspace_full_complex_np = kspace_full_complex.moveaxis(0, -1).cpu().numpy() # moved coil dim to last dim
    sens_maps = bart.bart(1, f'ecalib -d0 -m1', kspace_full_complex_np)
    return sens_maps

# def compute_sens_maps_np(masked_ksp):
#     ### compute sensitivity maps
#     masked_ksp = masked_ksp[...,0] + 1j*masked_ksp[...,1]
#     sens_maps = bart.bart(1, f'ecalib -d0 -m1', np.array([np.moveaxis(masked_ksp,0,2)]))
#     return np.moveaxis(sens_maps[0],2,0)

# MVUE are computed usig following bart params
def compute_sens_maps_np(masked_ksp):
    ### compute sensitivity maps
    masked_ksp = masked_ksp[...,0] + 1j*masked_ksp[...,1]
    sens_maps = bart.bart(1, f'ecalib -d0 -m1', np.array([np.moveaxis(masked_ksp,0,2)]))
    return np.moveaxis(sens_maps[0],2,0)

def compute_sens_maps_mp(masked_ksp, pool_size=8):
    # assume (Z, coils, Y, X, 2)
    #device = masked_ksp.get_device()
    iterates = list(masked_ksp.cpu().numpy()) if torch.is_tensor(masked_ksp) else list(masked_ksp)
    return np.array(mp.Pool(pool_size).map(compute_sens_maps_np, iterates))

def compute_l1_wavelet_solution(kspace, sensmaps, reg_param=4e-4):
    kspace_full_complex = torch.view_as_complex(kspace)
    kspace_full_complex_np = kspace_full_complex.moveaxis(0, -1).cpu().numpy()
    result_np = bart.bart(1, f'pics -l1 -r{reg_param}', kspace_full_complex_np, sensmaps.cpu().squeeze().numpy())
    return torch.view_as_real(torch.from_numpy(result_np)).to(kspace.device)

#def compute_sens_maps(masked_ksp):
    #### compute sensitivity maps
    #masked_ksp = masked_ksp[...,0] + 1j*masked_ksp[...,1]
    ## format ()
    #sens_maps = bart.bart(1, f'ecalib -d0 -m1', np.array([np.moveaxis(masked_ksp.detach().cpu().numpy(),0,2)]))
    ## C, Y, X -> 
    #return np.moveaxis(sens_maps[0],2,0)

#def compute_sens_maps_sigpy(masked_ksp):
    ## shape: (Nc, W, H, 2)
    #Nc, W, H, C = masked_ksp.shape
    #masked_ksp = masked_ksp[...,0] + 1j*masked_ksp[...,1]
    #masked_ksp_complex_np = torch.view_as_complex(masked_ksp).numpy()
    #sens_maps = mr.app.EspiritCalib(masked_ksp_complex_np)
    ## shape (H, W, Nc, 1)
    #return sens_maps.view(Nc, W, H, 1)

def vis_sense_maps(sens_maps):
    fig = plt.figure(figsize=(40,40))
    for i,s in enumerate(sens_maps):
        ax = fig.add_subplot(6,6,i+1)
        ax.imshow(np.abs(s),'gray')
        ax.set_title('coil {}'.format(i+1),fontsize=28)
        ax.axis('off')
    plt.show()
    return fig