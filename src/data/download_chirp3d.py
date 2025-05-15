import pandas as pd
import os
from tqdm import tqdm

if __name__ == "__main__":
    url = 'https://bridges.monash.edu/ndownloader/files/14206604'
    output_dir = '/media/ssd1//media/ssd1/chirp3d'
    os.system(f"wget -P {output_dir} {url}")