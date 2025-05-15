
from argparse import ArgumentParser
import numpy as np
import fastmri
from fastmri.data import transforms
from glob import glob
import h5py
from pathlib import Path
from tqdm import tqdm
import os
import ismrmrd
import ismrmrd.xsd

import logging

def ifftn(x, axes):
    # same order of shifts as in fastmri
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm='ortho'), axes=axes)

def read_ocmr(filename):
# Before running the code, install ismrmrd-python and ismrmrd-python-tools:
#  https://github.com/ismrmrd/ismrmrd-python
#  https://github.com/ismrmrd/ismrmrd-python-tools
# Last modified: 06-12-2020 by Chong Chen (Chong.Chen@osumc.edu)
#
# Input:  *.h5 file name
# Output: all_data    k-space data, orgnazide as {'kx'  'ky'  'kz'  'coil'  'phase'  'set'  'slice'  'rep'  'avg'}
#         param  some parameters of the scan
# 

# This is a function to read K-space from ISMRMD *.h5 data
# Modifid by Chong Chen (Chong.Chen@osumc.edu) based on the python script
# from https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/recon_ismrmrd_dataset.py

    if not os.path.isfile(filename):
        print("%s is not a valid file" % filename)
        raise SystemExit
    dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]

    # Matrix size
    eNx = enc.encodedSpace.matrixSize.x
    #eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z
    eNy = (enc.encodingLimits.kspace_encoding_step_1.maximum + 1); #no zero padding along Ny direction

    # Field of View
    eFOVx = enc.encodedSpace.fieldOfView_mm.x
    eFOVy = enc.encodedSpace.fieldOfView_mm.y
    eFOVz = enc.encodedSpace.fieldOfView_mm.z
    
    # Save the parameters    
    param = dict();
    param['TRes'] =  str(header.sequenceParameters.TR)
    param['FOV'] = [eFOVx, eFOVy, eFOVz]
    param['TE'] = str(header.sequenceParameters.TE)
    param['TI'] = str(header.sequenceParameters.TI)
    param['echo_spacing'] = str(header.sequenceParameters.echo_spacing)
    param['flipAngle_deg'] = str(header.sequenceParameters.flipAngle_deg)
    param['sequence_type'] = header.sequenceParameters.sequence_type

    # Read number of Slices, Reps, Contrasts, etc.
    nCoils = header.acquisitionSystemInformation.receiverChannels
    try:
        nSlices = enc.encodingLimits.slice.maximum + 1
    except:
        nSlices = 1
        
    try:
        nReps = enc.encodingLimits.repetition.maximum + 1
    except:
        nReps = 1
               
    try:
        nPhases = enc.encodingLimits.phase.maximum + 1
    except:
        nPhases = 1;

    try:
        nSets = enc.encodingLimits.set.maximum + 1;
    except:
        nSets = 1;

    try:
        nAverage = enc.encodingLimits.average.maximum + 1;
    except:
        nAverage = 1;   
        
    # TODO loop through the acquisitions looking for noise scans
    firstacq=0
    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)

        # TODO: Currently ignoring noise scans
        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            #print("Found noise scan at acq ", acqnum)
            continue
        else:
            firstacq = acqnum
            # print("Imaging acquisition starts acq ", acqnum)
            break

    # assymetry echo
    kx_prezp = 0;
    acq_first = dset.read_acquisition(firstacq)
    if  acq_first.center_sample*2 <  eNx:
        kx_prezp = eNx - acq_first.number_of_samples
         
    # Initialiaze a storage array
    param['kspace_dim'] = {'kx ky kz coil phase set slice rep avg'};
    all_data = np.zeros((eNx, eNy, eNz, nCoils, nPhases, nSets, nSlices, nReps, nAverage), dtype=np.complex64)
    
    # check if pilot tone (PT) is on
    pilottone = 0;
    try:
        if (header.userParameters.userParameterLong[3].name == 'PilotTone'):
            pilottone = header.userParameters.userParameterLong[3].value;
    except:
        pilottone = 0;  
            
    # if pilottone == 1:
        # print('Pilot Tone is on, discarding the first 3 and last 1 k-space point for each line')

    # Loop through the rest of the acquisitions and stuff
    for acqnum in range(firstacq,dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)
        if pilottone == 1: # discard the first 3 and last 1 k-space point to exclude PT artifact
            acq.data[:,[0,1,2,acq.data.shape[1]-1]] = 0        

        # Stuff into the buffer
        y = acq.idx.kspace_encode_step_1
        z = acq.idx.kspace_encode_step_2
        phase =  acq.idx.phase;
        set =  acq.idx.set;
        slice =  acq.idx.slice;
        rep =  acq.idx.repetition;
        avg = acq.idx.average;        
        all_data[kx_prezp:, y, z, :,phase, set, slice, rep, avg ] = np.transpose(acq.data)
        
    return all_data,param


def emulate_2D(kspace, device='cpu'):
    for i in range(kspace.shape[1]):
        kspace[:,i,:,:] = ifftn(kspace[:,i,:,:], axes=(0,)).astype(np.complex64)
    target = fastmri.ifft2c(transforms.to_tensor(kspace).to(device))
    target = fastmri.rss_complex(target, dim=-3).numpy().astype(np.float32)

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

    files = glob(os.path.join(dataset_dir, '*.h5'))
    for f in tqdm(files):
        try:
            kspace_full, _ = read_ocmr(f) # 'kspace_dim': {'kx ky kz coil phase set slice rep avg'}
            kspace_full = kspace_full[:,:,:,:,0,0,0,0,0]

            # sag
            kspace = kspace_full.transpose(0,3,1,2).copy()
            kspace, target = emulate_2D(kspace, device=device)
            save_file = os.path.join(save_dir, Path(f).stem + '_sag.h5')
            save_as_h5(save_file, kspace, target)

            # cor
            kspace = kspace_full.transpose(1,3,0,2).copy()
            kspace, target = emulate_2D(kspace, device=device)
            save_file = os.path.join(save_dir, Path(f).stem + '_cor.h5')
            save_as_h5(save_file, kspace, target)

            # ax
            kspace = kspace_full.transpose(2,3,0,1).copy()
            kspace, target = emulate_2D(kspace, device=device)
            save_file = os.path.join(save_dir, Path(f).stem + '_ax.h5')
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