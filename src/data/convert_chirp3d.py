
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


def ifftn(x, axes):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm='ortho'), axes=axes)

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
    
def cli_main(args):
    device = args.device
    dataset_dir = args.dataset_dir

    save_dir = os.path.join(dataset_dir, 'converted')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files = [
        os.path.join(dataset_dir, 'PLosOne_Chirp3d/mprage/kspace_Chirp_3DMPRAGE_256x256x192x17.h5'),
        os.path.join(dataset_dir, 'PLosOne_Chirp3d/mprage/kspace_Fourier_3DMPRAGE_256x256x192x17.h5'),
    ]

    for f in tqdm(files):
        try:
            with h5py.File(f, 'r') as hf:
                real = hf['ksData']['real'][()]
                imag = hf['ksData']['Imag'][()]
                kspace_3d = real + 1j*imag
                data = ifftn(kspace_3d, axes=(1,2,3))

            # axial
            img_data = data.transpose(2,0,3,1)
            kspace, target = emulate_2D(img_data, device=device)
            save_file = os.path.join(save_dir, Path(f).stem + '_ax.h5')
            save_as_h5(save_file, kspace, target)


            # sagittal
            img_data = data.transpose(1,0,2,3)
            kspace, target = emulate_2D(img_data, device=device)
            save_file = os.path.join(save_dir, Path(f).stem + '_sag.h5')
            save_as_h5(save_file, kspace, target)


            # coronal
            img_data = data.transpose(3,0,2,1)
            kspace, target = emulate_2D(img_data, device=device)
            save_file = os.path.join(save_dir, Path(f).stem + '_cor.h5')
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