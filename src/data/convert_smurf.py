
from argparse import ArgumentParser
import twixtools
import numpy as np
import fastmri
from fastmri.data import transforms
from matplotlib import pyplot as plt
from glob import glob
import h5py
from pathlib import Path
from tqdm import tqdm
import os

import logging


def fftn(x, axes):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm='ortho'), axes=axes)

def ifftn(x, axes):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm='ortho'), axes=axes)

def compute_target(kspace, shape=None, device='cuda'):
    kspace_torch = transforms.to_tensor(kspace).to(device)
    target_torch = fastmri.ifft2c(kspace_torch)
    target = fastmri.rss_complex(target_torch, dim=-3).cpu().numpy().astype(np.float32)
    if shape is not None:
        target = transforms.center_crop(target, shape)
    return target

def save_as_h5(filename, kspace, target, padding_left=None, padding_right=None):
    assert len(kspace.shape) == 4, 'kspace must have 4 dims: (slices, coils, readout, phase)'
    assert len(target.shape) == 3, 'target must have 3 dims: (slices, height, width)'
    kspace = kspace.astype(np.complex64)
    target = target.astype(np.float32)
    data = h5py.File(filename, 'w')
    data.create_dataset('kspace', data=kspace)
    data.create_dataset('reconstruction_rss', data=target)
    data.attrs.create('max', data=target.max())
    if padding_left is not None:
        data.attrs.create('padding_left', data=padding_left)
    if padding_right is not None:
        data.attrs.create('padding_right', data=padding_right)
    data.close()
    
def cli_main(args):
    device = args.device
    dataset_dir = args.dataset_dir

    save_dir = os.path.join(dataset_dir, 'converted')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files = glob(os.path.join(dataset_dir, '*.dat'))
    filtered_files = [f for f in files if 'SMURF' not in f and 'prescan' not in f]

    for f in tqdm(filtered_files):
        try:
            twix = twixtools.read_twix(f, verbose=False)

            # sort all 'imaging' mdbs into a k-space array
            image_mdbs = [mdb for mdb in twix[-1]['mdb'] if mdb.is_image_scan()]
            n_line = 1 + max([mdb.cLin for mdb in image_mdbs]) 
            n_slice = 1 + max([mdb.cSlc for mdb in image_mdbs]) 
            n_par = 1 + max([mdb.cPar for mdb in image_mdbs]) 
            n_set = 1 + max([mdb.cSet for mdb in image_mdbs]) 
            n_eco = 1 + max([mdb.cEco for mdb in image_mdbs])
            center_line = set([mdb.mdh.CenterLin for mdb in image_mdbs])
            if len(center_line) != 1:
                raise Exception("Multiple positions of center lines noted for this scan. Cannot proceed.")
            else:
                center_line = center_line.pop()

            # we use fastMRI convention where k-space matrix is tranposed
            # assume that all data were acquired with same number of channels & columns:
            n_channel, n_row = image_mdbs[0].data.shape

            kspace = np.zeros([n_eco,n_set, n_par, n_slice, n_channel, n_row, n_line], dtype=np.complex64)
            for mdb in  image_mdbs:
                kspace[mdb.cEco, mdb.cSet, mdb.cPar, mdb.cSlc, :, :, mdb.cLin] = mdb.data

            p_enc_size = int(twix[-1]['hdr']['Config']['PhaseEncodingLines'])
            r_enc_size = n_row // int(twix[-1]['hdr']['Config']['ReadoutOversamplingFactor'])
            kspace_pad = np.zeros((*kspace.shape[:-1], p_enc_size)).astype(np.complex64)
            padding_left = p_enc_size//2 - center_line
            padding_right = padding_left + n_line
            kspace_pad[..., padding_left:padding_right] = kspace
            kspace = kspace_pad.squeeze().astype(np.complex64)
            
            # If scan is 3D acquisition, emulate 2D by taking 1Difft across slice direction
            if n_par > 1 and n_slice == 1:
                kspace = ifftn(kspace, axes=(-4,))

            target = compute_target(kspace, shape=(r_enc_size, p_enc_size), device=device)

            if len(kspace.shape) > 5:
                raise Exception(f'k-space size exceeded {f}')

            if len(kspace.shape) == 5:
                for i in range(len(kspace)):
                    k = kspace[i]
                    t = target[i]
                    save_file = os.path.join(save_dir, Path(f).stem + '_m{:d}.h5'.format(i))
                    save_as_h5(save_file, k, t, padding_left=padding_left, padding_right=padding_right)
            else:
                save_file = os.path.join(save_dir, Path(f).stem + '.h5')
                save_as_h5(save_file, kspace, target, padding_left=padding_left, padding_right=padding_right)
            
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