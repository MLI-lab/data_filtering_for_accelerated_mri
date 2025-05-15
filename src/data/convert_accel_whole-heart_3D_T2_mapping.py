
from argparse import ArgumentParser
from glob import glob
import os
import h5py
import fastmri
from fastmri.data import transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm
import scipy.io

import logging

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

def ifftn(x, axes):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm='ortho'), axes=axes)

def cli_main(args):
    dataset_dir = args.dataset_dir
    device = args.device
    output_dir = os.path.join(dataset_dir, 'converted')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created directory: {output_dir}')
    
    files = glob(os.path.join(dataset_dir, '*.mat'))
    for fname in tqdm(files):
        try:
            data = scipy.io.loadmat(fname)

            kspace_full = data[Path(fname).stem]

            kspace_full = ifftn(kspace_full, axes = (2,))
            kspace_full = kspace_full.transpose(3,2,4,0,1).astype(np.complex64)

            kspace_full_torch = transforms.to_tensor(kspace_full)
            target_full = fastmri.ifft2c(kspace_full_torch.to(device)).cpu()
            target_full = fastmri.rss_complex(target_full, dim=-3).numpy().astype(np.float32)


            for i in range(0, len(kspace_full)):
                kspace = kspace_full[i]
                target = target_full[i]
                save_file = os.path.join(output_dir, Path(fname).stem + f'_m{i}' + '.h5')
                save_as_h5(save_file, kspace, target)

        except Exception as e:
            logger.error(f'Problem with converting file {fname}: {e}')
            print(f'Problem with converting file {fname}')


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