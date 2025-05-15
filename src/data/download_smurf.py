import pandas as pd
import os
from tqdm import tqdm

if __name__ == "__main__":
    url = 'https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/XNMCYI/'

    urls = []
    df = pd.read_csv('smurf.csv')
    df = df[df['Url'].str.contains('version=3.0')]
    for idn, fname in zip(df['Url'],df['Anchor Text']):
        print(idn)
        idn = idn.split('/')[-1].split('&')[0]
        urls.append((url+idn, fname))
    for url, fname in tqdm(urls):
        os.system(f"wget -O {os.path.join('/media/ssd1/smurf/', fname)} {url}")
