from argparse import ArgumentParser
import h5py
import numpy as np
from fastmri import fft2c, ifft2c, complex_abs
from fastmri.data import transforms as T
import torch
import json
import logging
from src.interfaces.config_models import BasePathConfigModel
from src.interfaces import config_factory
from src.interfaces.path_setups import import_bart_from_path_model
import multiprocessing as mp


def get_mvue(kspace, S):
    # calculate the coilwise images
    image  = ifft2c(kspace)
    # assert that the sensemaps are normalized (binary mask)
    S_norm = S.abs().square().sum(dim=1)
    S_binary = torch.where(S_norm > 0, torch.ones_like(S_norm), torch.zeros_like(S_norm))
    torch.testing.assert_close(S_norm, S_binary)
    # return the mvue
    return torch.sum(torch.view_as_complex(image) * torch.conj(S), dim=1)

def kspace_to_sensmaps_mvue(kspace):
    from bart import bart
    kspace = kspace[None,...].transpose(0,2,3,1)
    sens_maps = bart(1, "ecalib -m1 -d0", kspace)
    kspace = T.to_tensor(kspace)
    kspace = kspace.permute(0,3,1,2,4)
    sens_maps = torch.from_numpy(sens_maps.transpose(0,3,1,2))
    y = get_mvue(kspace, sens_maps)
    mvue = y.squeeze().cpu().numpy().astype(np.complex64)
    return sens_maps, mvue

def process_file(file, args):
    global files_remaining
    global files_remaining_lock

    logging.info(f"Start processing file {file}.")
    with h5py.File(file, 'r+') as data:
        mvue_exists = args.mvue_key in data.keys()
        smaps_exists = args.smaps_key in data.keys()

        # if args.overwrite or args.clear or (not mvue_exists and args.store_mvue) and (not smaps_exists and args.store_sensmaps):

        if  args.store_mvue and not mvue_exists or args.store_sensmaps and not smaps_exists or args.overwrite:
            kspace_all = data['kspace'][()]

            if args.parallel_pool_size_inner > 0:
                with mp.Pool(args.parallel_pool_size_inner) as p:
                    results = p.map(kspace_to_sensmaps_mvue, kspace_all)
            else:
                results = [kspace_to_sensmaps_mvue(kspace) for kspace in kspace_all]
                
            # unzip results
            sens_maps = np.stack([r[0][0] for r in results])
            reconstruction_mvue = np.stack([r[1] for r in results])

            maxval = np.abs(reconstruction_mvue).max()
            logging.info(f"Max value of mvue: {maxval}, shapes of sens_maps: {sens_maps.shape}, reconstruction_mvue: {reconstruction_mvue.shape}")


            if args.overwrite and mvue_exists and args.store_mvue:
                logging.info(f"Overwriting mvue file in {file}.")
                del data[args.mvue_key]
                data.create_dataset(args.mvue_key, data=reconstruction_mvue)
                data.attrs[args.mvue_max_key] = maxval
            elif not args.overwrite and not mvue_exists and args.store_mvue:
                logging.info(f"Creating mvue file in {file}.")
                data.create_dataset(args.mvue_key, data=reconstruction_mvue)
                data.attrs.create(args.mvue_max_key, data=maxval)

            if args.overwrite and smaps_exists and args.store_sensmaps:
                logging.info(f"Overwriting sensitivity_maps in {file}.")
                del data[args.smaps_key]
                data.create_dataset(args.smaps_key, data=sens_maps)
            elif not args.overwrite and not smaps_exists and args.store_sensmaps:
                logging.info(f"Creating sensitivity_maps in {file}.")
                data.create_dataset(args.smaps_key, data=sens_maps)

        else:
            print(f"Skip file {file}")

    with files_remaining_lock:
        files_remaining.value -= 1

    logging.info(f"Processed file {file}, remaining: {files_remaining.value}.")

def cli_main(args):
    logging.info(f"Validating base path setup: {args.path_config} and importing bart.")
    import_bart_from_path_model(config_factory.base_path_config_by_file(args.path_config))
    import bart

    path_to_json = args.dataset_path 
    with open(path_to_json) as f:
        json_data = json.load(f)

    files = [v['path'] for _, v in json_data['files'].items() if len(v['slices'].items()) > 0]
    total = len(files)

    global files_remaining
    global files_remaining_lock
    files_remaining = mp.Value('i', total)
    files_remaining_lock = mp.Lock()

    if args.parallel_pool_size > 0:

        with mp.Pool(args.parallel_pool_size) as p:
            p.starmap(process_file, [(f, args) for f in files])
    else:
        for i, file in enumerate(files):
            process_file(file, args)
            print(f"Processed {i+1}/{total}.")

def build_args():
    parser = ArgumentParser()

    # client arguments
    parser.add_argument(
        "-p",
        "--path_config", 
        type=str,
        required=True,
        help="Configuration of the base paths."
    )
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        help="Path to dataset json file.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing mvue/and sensmaps in h5 files.",
        default=False,
    )
    parser.add_argument(
        "-c",
        "--clear",
        action="store_true",
        help="Clear existing mvue/and sensmaps in h5 files (if sensmaps should not be saved anymore for example).",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--store_sensmaps",
        action="store_true",
        help="Store sensitivity maps in h5 files.",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--store_mvue",
        action="store_true",
        help="Store MVUE in h5 files.",
        default=False
    )
    parser.add_argument(
        "-k",
        "--store_kspace_attrs",
        action="store_true",
        help="Store kspace attributes in h5 volumes (norm).",
        default=False
    )
    parser.add_argument(
        "--mvue_key",
        type=str,
        help="Key for mvue in h5 files.",
        default='reconstruction_mvue'
    )
    parser.add_argument(
        "--mvue_max_key",
        type=str,
        help="Key for max mvue in h5 files.",
        default='max_mvue'
    )
    parser.add_argument(
        "--kspace_vol_norm_key",
        type=str,
        help="Key for kspace volume norm in h5 files.",
        default='kspace_vol_norm'
    )
    parser.add_argument(
        "--smaps_key",
        type=str,
        help="Key for sensitivity maps in h5 files.",
        default='sensitivity_maps'
    )
    parser.add_argument(
        "-l",
        "--log_file",
        type=str,
        default="temp_log.log",
        help="Path to log file (not logged to file if empty string)."
    )
    parser.add_argument(
        "--parallel_pool_size",
        type=int,
        default=4,
        help="Number of parallel processes for updating files."
    )
    parser.add_argument(
        "--parallel_pool_size_inner",
        type=int,
        default=0,
        help="Number of parallel processes for updating files."
    )
    args = parser.parse_args()
    return args

def run_cli():
    args = build_args()
    if args.log_file != "":
        logging.basicConfig(filename=args.log_file, level=logging.DEBUG, 
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    else:
        logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')

    cli_main(args)

if __name__ == "__main__":
    run_cli()