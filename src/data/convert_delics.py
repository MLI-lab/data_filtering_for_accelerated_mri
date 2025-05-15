
from argparse import ArgumentParser
import numpy as np
import fastmri
from fastmri.data import transforms
from glob import glob
import h5py
from pathlib import Path
from tqdm import tqdm
import os
import sigpy as sp
from scipy.io import loadmat

import logging


def emulate_2D(img_data, device='cpu'):
    target = transforms.to_tensor(img_data)
    kspace = transforms.tensor_to_complex_np(fastmri.fft2c(target.to(device)).cpu()).astype(np.complex64)
    target = fastmri.rss_complex(target, dim=1).numpy().astype(np.float32)

    return kspace, target

def save_as_h5(filename, kspace, target, padding_left=None, padding_right=None):
    assert len(kspace.shape) == 4, 'kspace must have 4 dims: (slices, coils, readout, phase)'
    assert len(target.shape) == 3, 'target must have 3 dims: (slices, height, width)'

    data = h5py.File(filename, 'w')
    data.create_dataset('kspace', data=kspace)
    data.create_dataset('reconstruction_rss', data=target)
    data.attrs.create('max', data=target.max())
    if padding_left is not None:
        data.attrs.create('padding_left', data=padding_left)
    if padding_right is not None:
        data.attrs.create('padding_right', data=padding_right)
    data.close()
    

def check_values(arr):
  assert np.sum(np.isnan(np.abs(arr.ravel()))) == 0, \
    ">>>>> Unexpected nan in array."
  assert np.sum(np.isinf(np.abs(arr.ravel()))) == 0, \
    ">>>>> Unexpected inf in array."

def compute_img_mrf(ksp_file, trj_file, dcf_file):
    trj = loadmat( trj_file)['k_3d'].transpose((1, 0, 2, 3)).astype(np.float32)
    check_values(trj)
    assert np.abs(trj.ravel()).max() < 0.5, \
    "Trajectory must be scaled between -1/2 and 1/2."

    dcf = np.load(dcf_file)

    ksp = np.load(ksp_file, mmap_mode='r')
    ksp = ksp.astype(np.complex64)

    # Remove rewinder points
    num_points = trj.shape[1]
    ksp = np.transpose(ksp[:num_points, ...], (1, 0, 2))

    # Split interleaves and time points
    ksp = np.reshape(ksp, (ksp.shape[0], ksp.shape[1], \
                            trj.shape[2], trj.shape[3]))

    ptt = 10
    ksp = ksp[:, ptt:, ...]

    trj = trj[:, ptt:, ...]

    check_values(trj)
    check_values(ksp)

    sx, sy, sz = 256, 256, 256
    trj[0, ...] *= sx
    trj[1, ...] *= sy
    trj[2, ...] *= sz

    trj = trj[::-1, ...].T
    ksp = np.transpose(ksp, (1, 2, 3, 0)).T


    dcf = dcf/np.linalg.norm(dcf.ravel(), np.inf)
    ksp = ksp * dcf
    ksp = ksp/np.linalg.norm(ksp)
    
    n_coils = ksp.shape[0]

    img = np.zeros((n_coils, sx, sy, sz)).astype(np.complex64)
    for i in range(n_coils):
        img[i] = sp.nufft_adjoint(ksp[i], trj, oshape=(sx, sy, sz))

    return img

def compute_img_gre(ksp_file):
    raw = np.load(ksp_file)
    ksp = np.reshape(raw, (raw.shape[0], raw.shape[1], 64, 64))
    ksp = np.transpose(ksp, (0, 2, 3, 1))
    tmp = np.zeros_like(ksp)
    tmp[:, :, 0::2, :] = ksp[:, :, :32, :]
    tmp[:, :, 1::2, :] = ksp[:, :, 32:, :]
    img = sp.ifft(tmp, axes=(0, 1))
    img = np.transpose(img, (2, 1, 0, 3))
    img = img[:, ::-1, ::-1, :]
    img = img.transpose(3,2,1,0)

    return img
    
def cli_main(args):
    device = args.device
    dataset_dir = args.dataset_dir

    save_dir = os.path.join(dataset_dir, 'train')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_files = glob(os.path.join(dataset_dir, 'data/**/raw*.npy'), recursive=True)
    gre_files = [f for f in all_files if 'gre' in f]
    two_min_mrf_files = [f for f in all_files if f not in gre_files and 'testing' in f and '000' not in f and '001' not in f]
    six_min_mrf_files = [f for f in all_files if f not in gre_files and f not in two_min_mrf_files]

    for f in tqdm(sorted(all_files)):
        try:
            ksp_file = f
            if ksp_file in gre_files:
                img  = compute_img_gre(ksp_file)

            elif ksp_file in two_min_mrf_files:
                trj_file = os.path.join(dataset_dir, 'data/shared/traj_grp16_inacc2.mat')
                dcf_file = os.path.join(dataset_dir, 'data/shared/dcf_2min.npy')
                img = compute_img_mrf(ksp_file, trj_file, dcf_file)

            elif ksp_file in six_min_mrf_files:
                trj_file = os.path.join(dataset_dir, 'data/shared/traj_grp48_inacc1.mat')
                dcf_file = os.path.join(dataset_dir, 'data/shared/dcf_6min.npy')
                img = compute_img_mrf(ksp_file, trj_file, dcf_file)
            
            save_name = Path(f).parents[1].stem + '_' + Path(f).parents[0].stem + '_' + Path(f).stem
            # ax 
            img_data = img.transpose(1,0,2,3)
            kspace, target = emulate_2D(img_data, device=device)
            save_file = os.path.join(save_dir, save_name + '_ax.h5')
            save_as_h5(save_file, kspace, target)

            # cor
            img_data = img.transpose(2,0,1,3)
            kspace, target = emulate_2D(img_data, device=device)
            save_file = os.path.join(save_dir, save_name + '_cor.h5')
            save_as_h5(save_file, kspace, target)

            # sag 
            img_data = img.transpose(3,0,1,2)
            kspace, target = emulate_2D(img_data, device=device)
            save_file = os.path.join(save_dir, save_name + '_sag.h5')
            save_as_h5(save_file, kspace, target)
        
        except Exception as e:
            logger.error(f'Problem with converting file {f}: {e}')
            print(f'Problem with converting file {f}')


def build_args():
    parser = ArgumentParser()

    # client arguments
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory where the original dataset is located",
    )

    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="device: cpu, cuda",
    )
    
    args = parser.parse_args()
    return args


def run_cli():
    args = build_args()
    cli_main(args)

if __name__ == "__main__":
    logging.basicConfig(filename='temp_log.log', level=logging.DEBUG, 
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger=logging.getLogger(__name__)
    run_cli()