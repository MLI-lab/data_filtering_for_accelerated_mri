
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
import sigpy as sp

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

def cli_main(args):
    device = args.device
    dataset_dir = args.dataset_dir

    save_dir_train = os.path.join(dataset_dir, 'converted/train')
    save_dir_test = os.path.join(dataset_dir, 'converted/test')
    if not os.path.exists(save_dir_train):
        os.makedirs(save_dir_train)
    if not os.path.exists(save_dir_test):
        os.makedirs(save_dir_test)

    files = glob(os.path.join(dataset_dir, '**/*.h5'), recursive=True)

    for f in tqdm(files):
        if Path(f).parent.stem == 'test':
            save_dir = save_dir_test
        else:
            save_dir = save_dir_train

        try:
            with h5py.File(f) as hf:
                kspace = hf['kspace'][()]
                kspace_tensor = transforms.to_tensor(kspace)
                target_tensor = fastmri.ifft2c(kspace_tensor.to(device)).cpu()

            # axial
            target_tensor_p = target_tensor.permute(3,1,2,0,4)

            kspace_p = transforms.tensor_to_complex_np(fastmri.fft2c(target_tensor_p.to(device)).cpu()).astype(np.complex64)
            target = fastmri.rss_complex(target_tensor_p, dim=1).numpy().astype(np.float32)
            save_file = os.path.join(save_dir, Path(f).stem[:-4] + '_ax.h5')
            save_as_h5(save_file, kspace_p, target)

            # coronal
            target_tensor_p = target_tensor.permute(2,1,3,0,4)

            kspace_p = transforms.tensor_to_complex_np(fastmri.fft2c(target_tensor_p.to(device)).cpu()).astype(np.complex64)
            target = fastmri.rss_complex(target_tensor_p, dim=1).numpy().astype(np.float32)
            save_file = os.path.join(save_dir, Path(f).stem[:-4] + '_cor.h5')
            save_as_h5(save_file, kspace_p, target)
            
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