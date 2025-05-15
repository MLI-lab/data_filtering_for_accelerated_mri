import os
from argparse import ArgumentParser


def convert_smurf(save_path):
    save_path = os.path.join(save_path, 'smurf')
    os.system(f"python src/data/convert_smurf.py --dataset_dir {save_path}")

def convert_extreme_mri(save_path):
    save_path = os.path.join(save_path, 'extreme_mri/')
    os.system(f"python src/data/convert_extreme-mri.py --dataset_dir {save_path}")
    
def convert_ocmr(save_path):
    save_path = os.path.join(save_path, 'ocmr')
    os.system(f"python src/data/convert_ocmr.py --dataset_dir {save_path}")

def convert_accel_whole_heart_3d_t2_mapping(save_path):
    save_path = os.path.join(save_path, 'accel_whole-heart_3D_T2_mapping')
    os.system(f"python src/data/convert_accel_whole-heart_3D_T2_mapping.py --dataset_dir {save_path}")

def convert_3dute(save_path):
    save_path = os.path.join(save_path, '3D-UTE_MRI')
    os.system(f"python src/data/convert_3d-ute --dataset_dir {save_path}")
    

def convert_fruits_phantom(save_path):
    save_path = os.path.join(save_path, 'fruits_phantom')
    os.system(f"python src/data/convert_fruits_phantom.py --dataset_dir {save_path}")

def convert_chirp3d(save_path):
    save_path = os.path.join(save_path, 'chirp3d')
    os.system(f"python src/data/convert_chirp3d.py --dataset_dir {save_path}")


def cli_main(args):
    save_path = args.save_path 
    convert_smurf(save_path)
    # convert_extreme_mri(save_path)
    # convert_ocmr(save_path)
    convert_accel_whole_heart_3d_t2_mapping(save_path)
    # convert_3dute(save_path)
    # convert_fruits_phantom(save_path)
    # convert_chirp3d(save_path)

if __name__ == "__main__":
    parser = ArgumentParser(description="parse datasets.")
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=True,
        help="Directory to save the parseed datasets.",
    )
    args = parser.parse_args()
    cli_main(args)

