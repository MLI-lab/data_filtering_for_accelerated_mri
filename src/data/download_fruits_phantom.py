import os
from glob import glob

if __name__ == "__main__":
    output_dir = '/media/ssd1//media/ssd1/fruits_phantom'
    os.system(f"zenodo_get -r 7509338 -o {output_dir}")

		
