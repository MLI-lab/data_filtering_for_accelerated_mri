from glob import glob
import os
import h5py
import uuid
from pathlib import Path
import json
import argparse

def create_json(data_dir, dataset_name, is_train, split):
    files = glob(os.path.join(data_dir, '*.h5'))

    num_slices = 0
    for fname in files: 
        with h5py.File(fname, "r") as hf:
            kspace = hf['kspace']
            num_slices += len(kspace)
    print(f'{dataset_name}, #slices: {num_slices}')

    data = {
        "dataset_name": dataset_name,
        "uuid": str(uuid.uuid4()),
        "num_slices": 0,
        "files": {}
    }

    for fname in files: 
        with h5py.File(fname, "r") as hf:
            kspace = hf['kspace']
        
            data['files'][Path(fname).name] = {}
            data['files'][Path(fname).name]['path'] = fname
            data['files'][Path(fname).name]["dataset_name"] = dataset_name
            data['files'][Path(fname).name]["split"] = split
            data['files'][Path(fname).name]["slices"] = {}
            for i in range(len(kspace)):
                data['files'][Path(fname).name]["slices"][i] = {}
                data['files'][Path(fname).name]["slices"][i]['freq'] = 1

    data['num_slices'] = num_slices
    data_json = json.dumps(data, indent=4)
    if is_train:
        with open(os.path.join('datasets/train', data['dataset_name'] + '.json'), 'w') as outfile:
            outfile.write(data_json)
    else:
        with open(os.path.join('datasets/evals/classic', data['dataset_name'] + '.json'), 'w') as outfile:
            outfile.write(data_json)


def run_cli_main(save_path):
    # Read as: dataset name: {offical_split (if exist mentioned by the original data source): [path, is_train]}
    datasets_splits = {
        'smurf': {'test': [os.path.join(save_path, 'smurf', 'converted'), False]},
        'accel_whole-heart_3D_T2_mapping': {'train': [os.path.join(save_path, 'accel_whole-heart_3D_T2_mapping', 'converted'), True]},
    }

    for dataset_name, splits in datasets_splits.items():
        for split, (data_dir, is_train) in splits.items():
            create_json(data_dir, dataset_name, is_train, split)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset JSON files.")
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=True,
        help="Directory where the datasets are stored.",
    )
    args = parser.parse_args()
    run_cli_main(args.save_path)
