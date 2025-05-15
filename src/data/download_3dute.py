import pandas as pd
import os
from tqdm import tqdm

if __name__ == "__main__":
    url = 'https://dataverse.harvard.edu/api/access/datafile/'

    urls = []
    df = pd.read_csv('3D-UTE_MRI.csv')
    df = df[df['Url'].str.contains('version=1.0')]
    for idn, fname in zip(df['Url'],df['Anchor Text']):
        idn = idn.split('=')[1].split('&')[0]
        urls.append((url+idn, fname))
        print(url+idn)
    for url, fname in tqdm(urls):
        os.system(f"wget -O {os.path.join('/media/ssd1/3D-UTE_MRI', fname)} {url}")
