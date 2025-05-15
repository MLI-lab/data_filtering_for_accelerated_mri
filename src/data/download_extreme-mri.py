import os

if __name__ == "__main__":
    dce_output_dir = '/media/ssd1//media/ssd1/extreme_mri/dce_3d_cone'
    lung_output_dir = '/media/ssd1//media/ssd1/extreme_mri/lung_3d_ute_radial'
    os.system(f"zenodo_get -r 4048824 -o {dce_output_dir}")
    os.system(f"zenodo_get -r 4048817 -o {lung_output_dir}")


		
