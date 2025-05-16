import h5py
from argparse import ArgumentParser
import pathlib
import numpy as np
import glob
import os
from tqdm.autonotebook import tqdm

# Functions to generate reconstruction target
def ifft2_np(x):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1)) 

# Functions to generate reconstruction target
def fft2_np(x):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1)) 

def rss_np(x, axis=1):
    return np.sqrt(np.sum(np.square(np.abs(x)), axis=axis))

def cli_main(args):
    print(args)
    if not os.path.exists(args.output_dir):
        print('Creating output directory...')
        os.makedirs(args.output_dir)
        
    folder = list(pathlib.Path(args.input_dir).glob('*.h5'))
            
    for fname in tqdm(folder, total=len(folder)):
        with h5py.File(fname, 'r') as hf:
            kspace = hf['kspace'][()]
            kspace = kspace.transpose(-2,0,-1,1,2) # Echo, Slice, Coil, Height, Width

            target = ifft2_np(kspace)
            target = target.transpose(0,4,2,3,1) # sagittal view
            target = target/1e11 # similar range to fastmri
            kspace = fft2_np(target)
            target = rss_np(target, axis=2)
            
            kspace = kspace.astype(np.complex64)
            target = target.astype(np.float32)
            
            k_echo_1 = kspace[0]
            t_echo_1 = target[0]

            save_file = os.path.join(args.output_dir, 'E1_' + str(fname.name))
            data = h5py.File(save_file, 'w')
            data.create_dataset('kspace', data=k_echo_1)
            data.create_dataset('reconstruction_rss', data=t_echo_1)
            data.attrs.create('max', data=t_echo_1.max())
            padding_left = 48
            padding_right = 464
            data.attrs.__setitem__('padding_left', padding_left)
            data.attrs.__setitem__('padding_right', padding_right)
            data.close()

            k_echo_2 = kspace[1]
            t_echo_2 = target[1]

            save_file = os.path.join(args.output_dir, 'E2_' + str(fname.name))
            data = h5py.File(save_file, 'w')
            data.create_dataset('kspace', data=k_echo_2)
            data.create_dataset('reconstruction_rss', data=t_echo_2)
            data.attrs.create('max', data=t_echo_2.max())
            padding_left = 48
            padding_right = 464
            data.attrs.__setitem__('padding_left', padding_left)
            data.attrs.__setitem__('padding_right', padding_right)
            data.close()
        
def build_args():
    parser = ArgumentParser()

    # client arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory where the original dataset is located.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save the converted dataset.",
    )
    
    args = parser.parse_args()
    return args
    
def run_cli():
    args = build_args()
    # run conversion
    cli_main(args)

if __name__ == "__main__":
    run_cli()