"""
Converts the original prostate fastMRI dataset to a format that is more consistent with
the original fastMRI brain and knee dataset.

1. the 3 'averages' are combined into one kspace by averaging the two 'averages' corresponding
to the odd kspace lines and filling the even kspace lines with 'average' corresposding to the
even kspace lines.

2. kspace is then padded according to ismrmrd header file

3. a new target is then created by applying ifft to the modifed kspace and center cropping

modified kspace and target are then saved as a new .h5 file
"""
import os
import numpy as np
import pathlib
import h5py
from tqdm.autonotebook import tqdm
from argparse import ArgumentParser
import xml.etree.ElementTree as etree
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)

def pad_kspace(kspace, ismrmrd_xml_header):
    et_root = etree.fromstring(ismrmrd_xml_header)
    enc = ["encoding", "encodedSpace", "matrixSize"]
    enc_size = (
        int(et_query(et_root, enc + ["x"])),
        int(et_query(et_root, enc + ["y"])),
        int(et_query(et_root, enc + ["z"])),
    )
    lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
    enc_limits_center = int(et_query(et_root, lims + ["center"]))
    enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

    padding_left = enc_size[1] // 2 - enc_limits_center
    padding_right = padding_left + enc_limits_max

    padded_kspace = np.zeros((kspace.shape[0], kspace.shape[1], enc_size[0], enc_size[1])).astype(np.complex64)
    padded_kspace[...,padding_left:padding_right] = kspace
    
    return padded_kspace

# Functions to generate reconstruction target
def ifft2_np(x):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1)) 

def kspace_to_target(x):
    return np.sqrt(np.sum(np.square(np.abs(ifft2_np(x))), axis=-3))

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

# Main conversion code
def cli_main(args):
    print(args)
    if not os.path.exists(args.output_dir):
        print('Creating output directory...')
        os.makedirs(args.output_dir)
        
    mri_files = list(pathlib.Path(args.input_dir).glob('*.h5'))
    print(mri_files)
    for i, mri_f in tqdm(zip(range(len(mri_files)), sorted(mri_files)), total=len(mri_files)):
        with h5py.File(mri_f, "r") as hf:
            k = hf['kspace'][()]
            kspace = np.zeros_like(k[0])
            kspace[...,1::2] = (k[0,...,1::2] + k[2,...,1::2]) / 2
            kspace[...,::2] = k[1,...,::2]
            
            # Zero padding kspace if needed
            xml_header = hf['ismrmrd_header'][()]
            kspace = pad_kspace(kspace, xml_header)
            target = kspace_to_target(kspace)
            target = center_crop(target, (320,320))

            target = target.astype(np.float32)
            kspace = kspace.astype(np.complex64)
            
            save_file = os.path.join(args.output_dir, str(mri_f.name))
            data = h5py.File(save_file, 'w')
            data.create_dataset('ismrmrd_header', data=xml_header)
            data.create_dataset('kspace', data=kspace)
            data.create_dataset('reconstruction_rss', data=target)
            data.attrs.__setitem__('acquisition', hf.attrs.__getitem__('acquisition'))
            data.attrs.__setitem__('max', target.max())
            data.attrs.__setitem__('norm', np.linalg.norm(target))
            data.attrs.__setitem__('patient_id', hf.attrs.__getitem__('patient_id'))  
            data.close()
        
    print('Finished.')
    
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