import os
from glob import glob

if __name__ == "__main__":
    output_dir = '/media/ssd1//media/ssd1/delics'
    os.system(f"wget -P {output_dir} https://zenodo.org/records/7734431/files/shared.tar.gz?download=1")
    os.system(f"zenodo_get -d 10.5281/zenodo.7703200 -o {output_dir}")
    os.system(f"zenodo_get -d 10.5281/zenodo.7697373 -o {output_dir}")

    files = glob(os.path.join(output_dir, '*.tar.gz'))
    for f in files:
        os.system(f"tar -xzvf {f} -C {output_dir}")	

		
