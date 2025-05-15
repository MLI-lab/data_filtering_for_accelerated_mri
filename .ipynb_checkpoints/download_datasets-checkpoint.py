import pandas as pd
import os
from tqdm import tqdm
from argparse import ArgumentParser
import zipfile


def download_smurf(save_path):
    url = 'https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/XNMCYI/'
    urls = []
    df = pd.read_csv('src/data/smurf.csv')
    df = df[df['Url'].str.contains('version=3.0')]
    for idn, fname in zip(df['Url'],df['Anchor Text']):
        idn = idn.split('/')[-1].split('&')[0]
        urls.append((url+idn, fname))
    save_path = os.path.join(save_path, 'smurf')
    os.makedirs(save_path, exist_ok=True)
    for url, fname in tqdm(urls):
        os.system(f"wget -O {os.path.join(save_path, fname)} {url}")

def download_extreme_mri(save_path):
    save_path_1 = os.path.join(save_path, 'extreme_mri/dce_3d_cone')
    os.makedirs(save_path_1, exist_ok=True)
    os.system(f"zenodo_get -r 4048824 -o {save_path_1}")
    save_path_2 = os.path.join(save_path, 'extreme_mri/lung_3d_ute_radial')
    os.makedirs(save_path_2, exist_ok=True)
    os.system(f"zenodo_get -r 4048817 -o {save_path_2}")
    
def download_ocmr(save_path):
    url = 'https://ocmr.s3.amazonaws.com/data/ocmr_cine.tar.gz'
    save_path = os.path.join(save_path, 'ocmr')
    os.system(f"wget -P {save_path} {url}")
    os.system(f"tar -xvf ocmr_cine.tar.gz -C {save_path}")

def download_accel_whole_heart_3d_t2_mapping(save_path):
    url = 'https://lifesciences.datastations.nl/api/access/datafile/'
    urls = []
    df = pd.read_csv('src/data/accel_whole-heart_3D_T2_mapping.csv')
    df = df[df['Url'].str.contains('version=1.0')]
    for idn, fname in zip(df['Url'],df['Anchor Text']):
        idn = idn.split('=')[1].split('&')[0]
        urls.append((url+idn, fname))
    save_path = os.path.join(save_path, 'accel_whole-heart_3D_T2_mapping')
    os.makedirs(save_path, exist_ok=True)
    for url, fname in tqdm(urls):
        os.system(f"wget -O {os.path.join(save_path, fname)} {url}")

def download_3dute(save_path):
    url = 'https://dataverse.harvard.edu/api/access/datafile/'
    urls = []
    df = pd.read_csv('src/data/3D-UTE_MRI.csv')
    df = df[df['Url'].str.contains('version=1.0')]
    for idn, fname in zip(df['Url'],df['Anchor Text']):
        idn = idn.split('=')[1].split('&')[0]
        urls.append((url+idn, fname))
        print(url+idn)
    save_path = os.path.join(save_path, '3D-UTE_MRI')
    os.makedirs(save_path, exist_ok=True)
    for url, fname in tqdm(urls):
        os.system(f"wget -O {os.path.join(save_path, fname)} {url}")

def download_fruits_phantom(save_path):
    save_path = os.path.join(save_path, 'fruits_phantom')
    os.makedirs(save_path, exist_ok=True)
    os.system(f"zenodo_get -r 7509338 -o {save_path}")

# def download_chirp3d(save_path):
#     url = 'https://bridges.monash.edu/ndownloader/files/14206604'
#     save_path = os.path.join(save_path, 'chirp3d')
#     os.makedirs(save_path, exist_ok=True)
#     os.system(f"wget -P {save_path} {url}")
#     os.system(f"7z e {os.path.join(save_path, '14206604')} -o{save_path} -y")


def cli_main(args):
    save_path = args.save_path 
    download_smurf(save_path)
    # download_extreme_mri(save_path)
    # download_ocmr(save_path)
    download_accel_whole_heart_3d_t2_mapping(save_path)
    # download_3dute(save_path)
    # download_fruits_phantom(save_path)
    # download_chirp3d(save_path)

if __name__ == "__main__":
    parser = ArgumentParser(description="Download datasets.")
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=True,
        help="Directory to save the downloaded datasets.",
    )
    args = parser.parse_args()
    cli_main(args)

