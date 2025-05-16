import os
import h5py 
import torch
import ismrmrd
import xmltodict as xd
import numpy as np
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
from ismrmrd import Dataset as read_ismrmrd
from ismrmrd.xsd import CreateFromDocument as parse_ismrmd_header
from fastmri import fft2c, ifft2c, rss_complex
from fastmri.data.transforms import to_tensor, tensor_to_complex_np
from scipy.ndimage import fourier_shift
from argparse import ArgumentParser
from typing import List, Optional
import pathlib
import torch
import torch.fft

# Functions to generate reconstruction target
def ifft3_np(x):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=(-3,-2,-1)), axes=(-3,-2,-1), norm='ortho'), axes=(-3,-2,-1))

def ifft2_np(x):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1)) 

def fft2_np(x):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1)) 

def rss_np(x):
    return np.sqrt(np.sum(np.square(np.abs(x)), axis=-3))

def fft3c(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-4, -3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-3, -2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-4, -3, -2])

    return data


def ifft3c(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-4, -3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-3, -2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-4, -3, -2])

    return data

def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)
    
def cli_main(args):
    if not os.path.exists(args.output_dir):
        print('Creating output directory...')
        os.makedirs(args.output_dir)
    output_dir = args.output_dir
    device = args.device
    mri_files = list(pathlib.Path(args.input_dir).glob('*.h5'))
    for i, file in tqdm(zip(range(len(mri_files)), sorted(mri_files)), total=len(mri_files)):
        dset = read_ismrmrd(file, 'dataset')
        header = parse_ismrmd_header(dset.read_xml_header())
        nX = header.encoding[0].encodedSpace.matrixSize.x
        nY = header.encoding[0].encodedSpace.matrixSize.y
        nZ = header.encoding[0].encodedSpace.matrixSize.z
        nCoils = header.acquisitionSystemInformation.receiverChannels
        kspace = np.zeros((nCoils, nX, nY, nZ), dtype=np.complex64)
        n = dset.number_of_acquisitions()
    
        for i in range(n):
            acq = dset.read_acquisition(i)
            i_ky = acq.idx.kspace_encode_step_1
            i_kz = acq.idx.kspace_encode_step_2
            kspace[:,:,i_ky,i_kz] = acq.data
        dset.close()
    
        kspace = kspace / 1e14
        kspace = to_tensor(kspace).to(device)
        target_t = ifft3c(kspace)
        
        # Sagittal
        target = target_t.permute(1,0,3,2,4)
        # target = torch.flip(target, (-3, ))
        kspace = fft2c(target)
        target = rss_complex(target, 1).cpu().numpy().astype(np.float32)
        kspace = tensor_to_complex_np(kspace.cpu()).astype(np.complex64)
    
        if not os.path.exists(output_dir+'sag/'):
            print('Creating output directory...')
            os.makedirs(output_dir+'sag/')
    
        save_file = os.path.join(output_dir+'sag/', file.split('/')[-1])
        data = h5py.File(save_file, 'w')
        data.create_dataset('kspace', data=kspace)
        data.create_dataset('reconstruction_rss', data=target)
        data.attrs.create('max', data=target.max())
        data.close()
    
        # Coronal
        target = target_t.permute(2,0,3,1,4)
        # target = torch.flip(target, (-3, ))
        kspace = fft2c(target)
        target = rss_complex(target, 1).cpu().numpy().astype(np.float32)
        kspace = tensor_to_complex_np(kspace.cpu()).astype(np.complex64)
    
        if not os.path.exists(output_dir+'cor/'):
            print('Creating output directory...')
            os.makedirs(output_dir+'cor/')
            
        save_file = os.path.join(output_dir+'cor/', file.split('/')[-1])
        data = h5py.File(save_file, 'w')
        data.create_dataset('kspace', data=kspace)
        data.create_dataset('reconstruction_rss', data=target)
        data.attrs.create('max', data=target.max())
        data.close()
        
        # Axial
        target = target_t.permute(3,0,2,1,4)
        # target = torch.flip(target, (-3, ))
        kspace = fft2c(target)
        target = rss_complex(target, 1).cpu().numpy().astype(np.float32)
        kspace = tensor_to_complex_np(kspace.cpu()).astype(np.complex64)
    
        if not os.path.exists(output_dir+'ax/'):
            print('Creating output directory...')
            os.makedirs(output_dir+'ax/')
            
        save_file = os.path.join(output_dir+'ax/', file.split('/')[-1])
        data = h5py.File(save_file, 'w')
        data.create_dataset('kspace', data=kspace)
        data.create_dataset('reconstruction_rss', data=target)
        data.attrs.create('max', data=target.max())
        data.close()
    
def build_args():
    parser = ArgumentParser()

    # client arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory where the original data is located",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save the converted dataset.",
    )

    parser.add_argument(
        "--device",
        type=str,
        help="device: cpu, cuda",
    )
    
    args = parser.parse_args()
    return args


def run_cli():
    args = build_args()
    # run conversion
    cli_main(args)

if __name__ == "__main__":
    run_cli()










