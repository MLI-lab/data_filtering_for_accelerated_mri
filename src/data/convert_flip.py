import h5py
from argparse import ArgumentParser
import pathlib
import numpy as np
import glob
import os
from tqdm.autonotebook import tqdm
from scipy.ndimage import fourier_shift

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

def center_crop(data, shape):
    """
    Adjusted code from fastMRI github repository
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape
    Returns:
        The center cropped image.
    """
    assert shape[0] > 0 and shape[1] > 0, "Desired shape of ({:d},{:d}) invalid. shape must contain positive integers".format(*shape)
    w_from = (data.shape[-2] - shape[0]) // 2
    w_from = w_from if w_from > 0 else 0
    
    h_from = (data.shape[-1] - shape[1]) // 2
    h_from = h_from if h_from > 0 else 0

    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

def cli_main(args):
    print(args)
    if not os.path.exists(args.output_dir):
        print('Creating output directory...')
        os.makedirs(args.output_dir)
        
    folder = list(pathlib.Path(args.input_dir).glob('*.h5'))
            
    for fname in tqdm(folder, total=len(folder)):
        with h5py.File(fname, 'r') as hf:
            kspace = hf['kspace'][()]
            target = hf['reconstruction_rss'][()]
            _, h, w = target.shape
            
            target = np.flip(target, axis=-2)
            kspace = np.flip(kspace, axis=-2)
            kspace = fourier_shift(kspace, (0,0,-1,0))
            
            kspace = kspace.astype(np.complex64)
            target = target.astype(np.float32)

            save_file = os.path.join(args.output_dir, str(fname.name))
            data = h5py.File(save_file, 'w')
            if 'ismrmrd_header' in list(hf.keys()):
                data.create_dataset('ismrmrd_header', data=hf['ismrmrd_header'][()])
            data.create_dataset('kspace', data=kspace)
            data.create_dataset('reconstruction_rss', data=target)
            data.attrs.__setitem__('max', hf.attrs.__getitem__('max'))
            if 'acquisition' in dict(hf.attrs):
                data.attrs.__setitem__('acquisition', hf.attrs.__getitem__('acquisition'))
            if 'patient_id' in dict(hf.attrs):
                data.attrs.__setitem__('patient_id', hf.attrs.__getitem__('patient_id'))
            if 'padding_left' in dict(hf.attrs):
                data.attrs.__setitem__('padding_left', hf.attrs.__getitem__('padding_left'))
            if 'padding_right' in dict(hf.attrs):
                data.attrs.__setitem__('padding_right', hf.attrs.__getitem__('padding_right'))

            data.close()

            # os.remove(fname)
        
def build_args():
    parser = ArgumentParser()

    # client arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory where the dataset is located.",
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