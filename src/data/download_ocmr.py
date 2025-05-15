import os

if __name__ == "__main__":
    url = 'https://ocmr.s3.amazonaws.com/data/ocmr_cine.tar.gz'
    output_dir = '/media/ssd1//media/ssd1/ocmr'
    os.system(f"wget -P {output_dir} {url}")
    os.system(f"tar -xvf ocmr_cine.tar.gz -C {output_dir}")