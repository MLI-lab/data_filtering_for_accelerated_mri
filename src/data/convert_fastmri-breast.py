
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


def get_traj(N_spokes=13, N_time=1, base_res=320, gind=1):

    N_tot_spokes = N_spokes * N_time

    N_samples = base_res * 2

    base_lin = np.arange(N_samples).reshape(1, -1) - base_res

    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)

    base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((N_tot_spokes, N_samples, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    traj = traj / 2

    traj = traj.reshape(N_time, N_spokes, N_samples, 2)

    return np.squeeze(traj)


def grid_recon(ksp_file):
    images_per_slab =192
    center_partition = 31
    spokes_per_frame = 288
    slice_inc = 1
    slice_idx = 0


    ksp_f = ksp_file.T
    ksp_f = np.transpose(ksp_f, (4, 3, 2, 1, 0))


    ksp = ksp_f[0] + 1j * ksp_f[1]
    ksp = np.transpose(ksp, (3, 2, 0, 1))

    # zero-fill the slice dimension
    partitions = ksp.shape[0]
    shift = int(images_per_slab / 2 - center_partition)

    ksp_zf = np.zeros_like(ksp, shape=[images_per_slab] + list(ksp.shape[1:]))
    ksp_zf[shift : shift + partitions, ...] = ksp

    ksp_zf = sp.fft(ksp_zf, axes=(0,))

    N_slices, N_coils, N_spokes, N_samples = ksp_zf.shape

    base_res = N_samples // 2

    N_time = N_spokes // spokes_per_frame

    N_spokes_prep = N_time * spokes_per_frame

    ksp_redu = ksp_zf[:, :, :N_spokes_prep, :]

    # retrospecitvely split spokes
    ksp_prep = np.swapaxes(ksp_redu, 0, 2)
    ksp_prep_shape = ksp_prep.shape
    ksp_prep = np.reshape(ksp_prep, [N_time, spokes_per_frame] + list(ksp_prep_shape[1:]))
    ksp_prep = np.transpose(ksp_prep, (3, 0, 2, 1, 4))
    ksp_prep = ksp_prep[:, :, None, :, None, :, :]

    # trajectories
    traj = get_traj(N_spokes=spokes_per_frame,
                    N_time=N_time, base_res=base_res,
                    gind=1)

    dcf = (traj[..., 0]**2 + traj[..., 1]**2)**0.5
    img = sp.nufft_adjoint(ksp_prep.astype(np.complex64)*dcf.astype(np.float32), traj.astype(np.float32),oshape=(192,16,320,320))

    return img 
    
def cli_main(args):
    device = args.device
    dataset_dir = args.dataset_dir

    save_dir_train = os.path.join(dataset_dir, 'converted/train')
    save_dir_test = os.path.join(dataset_dir, 'converted/test')
    if not os.path.exists(save_dir_train):
        os.makedirs(save_dir_train)
    if not os.path.exists(save_dir_test):
        os.makedirs(save_dir_test)

    files = glob(os.path.join(dataset_dir, '[!converted]**/*.h5'), recursive=True)
    test_files = open("fastmri_beast_test.txt", "r").read().split("\n")
    for f in tqdm(files):
            if Path(f).stem[:-2] in test_files:
                save_dir = save_dir_test
            else:
                save_dir = save_dir_train

            try:
                with h5py.File(f, 'r') as hf:
                    ksp_file = hf['kspace'][:]
                    data = grid_recon(ksp_file)

                # axial
                img_data = data.transpose(0,1,2,3)
                kspace, target = emulate_2D(img_data, device=device)
                save_file = os.path.join(save_dir, Path(f).stem + '_ax.h5')
                save_as_h5(save_file, kspace, target)

                if Path(f).stem[:-2] not in test_files:
                    # sagittal
                    img_data = data.transpose(3,1,2,0)
                    kspace, target = emulate_2D(img_data, device=device)
                    save_file = os.path.join(save_dir, Path(f).stem + '_sag.h5')
                    save_as_h5(save_file, kspace, target)


                    # coronal
                    img_data = data.transpose(2,1,3,0)
                    kspace, target = emulate_2D(img_data, device=device)
                    save_file = os.path.join(save_dir, Path(f).stem + '_cor.h5')
                    save_as_h5(save_file, kspace, target)

                os.remove(f)
                
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