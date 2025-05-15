from glob import glob
import os

if __name__ == "__main__":
    train_files = glob('datasets/train/*.json')
    test_files = glob('datasets/evals/classic/*.json')
    train_files = [f for f in train_files if 'accel_whole-heart_3D_T2_mapping' in f] # for demo
    test_files = [f for f in train_files if 'smurf' in f] # for demo
    for f in train_files:
        os.system(f"python add_mvue_smaps_to_h5.py -p configs/paths/output_dirs.yml -d {f} -m")
    for f in test_files:
        os.system(f"python add_mvue_smaps_to_h5.py -p configs/paths/output_dirs.yml -d {f} -m -s")

