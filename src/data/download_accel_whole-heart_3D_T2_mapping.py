import pandas as pd
import os
from tqdm import tqdm

if __name__ == "__main__":
    url = 'https://lifesciences.datastations.nl/api/access/datafile/'

    urls = []
    df = pd.read_csv('accel_whole-heart_3D_T2_mapping.csv')
    df = df[df['Url'].str.contains('version=1.0')]
    for idn, fname in zip(df['Url'],df['Anchor Text']):
        idn = idn.split('=')[1].split('&')[0]
        urls.append((url+idn, fname))
    for url, fname in tqdm(urls):
        os.system(f"wget -O {os.path.join('/media/ssd1/accel_whole-heart_3D_T2_mapping', fname)} {url}")
