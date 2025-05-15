
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

import logging


def emulate_2D(img_data, device='cpu'):
    target = transforms.to_tensor(img_data)
    kspace = transforms.tensor_to_complex_np(fastmri.fft2c(target.to(device)).cpu()).astype(np.complex64)
    target = fastmri.rss_complex(target, dim=1).numpy().astype(np.float32)

    return kspace, target

def gridding_recon(ksp, coord, dcf, T=1, device=sp.cpu_device):
    device = sp.Device(device)
    xp = device.xp
    num_coils, num_tr, num_ro = ksp.shape
    tr_per_frame = num_tr // T
    img_shape = sp.estimate_shape(coord)

    with device:
        img = []
        for t in range(T):
            tr_start = t * tr_per_frame
            tr_end = (t + 1) * tr_per_frame
            coord_t = sp.to_device(
                coord[tr_start:tr_end], device)
            dcf_t = sp.to_device(dcf[tr_start:tr_end], device)

            img_t = []
            for c in range(num_coils):
                ksp_tc = sp.to_device(ksp[c, tr_start:tr_end, :], device)

                img_t.append(sp.nufft_adjoint(
                    ksp_tc * dcf_t, coord_t, img_shape))
                
            img_t = np.stack(img_t)
            img.append(sp.to_device(img_t))

    img = np.stack(img)
    return img

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

    files_dce = os.path.join(dataset_dir, 'dce_3d_cone')
    files_lung = os.path.join(dataset_dir, 'lung_3d_ute_radial')
    files = [files_dce, files_lung]
    for f in tqdm(files):
        try:
            kspace = np.load(os.path.join(f, 'ksp.npy'))
            dcf = np.load(os.path.join(f, 'dcf.npy'))
            coord = np.load(os.path.join(f, 'coord.npy'))

            img = gridding_recon(kspace, coord, dcf)[0]

            save_name = Path(f).stem

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