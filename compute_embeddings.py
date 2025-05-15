import os
from dreamsim import dreamsim
from math import floor, ceil
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
from glob import glob
from src.train_eval_setups.end_to_end.utils import SliceDataset, natural_sort
from fastmri.data.transforms import center_crop, to_tensor
from tqdm import tqdm
from skimage.util.shape import view_as_blocks
import argparse


def pad(x, size):

    h, w = x.shape
    hp, wp = size, size
    f1 = ( (wp - w % wp) % wp ) / 2
    f2 = ( (hp - h % hp) % hp ) / 2
    wpad = (floor(f1), ceil(f1))
    hpad = (floor(f2), ceil(f2))
    x = np.pad(x, (hpad, wpad), 'constant', constant_values=(0, 0))
    
    return x
    
def data_transform(kspace, mask, target, attrs, fname, dataslice):
    maxval = attrs['max']
    target = np.abs(target) / maxval
    crop_size = 128 # dreamsim is 224    

    # non-overlaping patches
    target = pad(target, crop_size)
    crops = view_as_blocks(target, block_shape=(crop_size, crop_size)).reshape(-1,crop_size,crop_size)
    
    n_crops, h, w = crops.shape
    target_tensor = to_tensor(crops).unsqueeze(1).expand(n_crops,3,h,w).to(torch.float)
    fname = n_crops*[fname]
    dataslice = torch.tensor(n_crops*[dataslice])
    return target_tensor, fname, dataslice

def compute_embeddings(save_path, dataset_json, model):
    with open(dataset_json) as f:
        dataset_name = json.load(f)['dataset_name']

    save_path_batches = os.path.join(save_path, 'dreamsim_ensemble_emb_p128_'+dataset_name)
    os.makedirs(save_path_batches, exist_ok=True)

    dataset = SliceDataset(dataset_json, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

    i = 0

    for image, fname, dataslice in tqdm(dataloader):
        b,n,c,h,w = image.shape
        image = image.view(b*n,c,h,w)
        emb = model.embed(image.to(model.device)).to(torch.float16).cpu()
        fname = list(zip(*fname))
        dataslice = dataslice
        data = {
            'emb': emb,
            'fname' : fname,
            'dataslice' : dataslice
        }
        i += 1
        torch.save(data, os.path.join(save_path_batches, f'batch_{i}.pt'))

    files = glob(os.path.join(save_path_batches, f'*.pt'))
    embs = []
    fnames = []
    dataslices = []
    for f in tqdm(files):
        data = torch.load(f)
        embs.append(data['emb'])
        fnames += data['fname']
        dataslices += data['dataslice']

    embs = torch.concatenate(embs)
    fnames = [d for f in fnames for d in f]
    dataslices = torch.concatenate(dataslices)

    embs_datapool = {
        'embs': embs,
        'fnames': fnames,
        'slices': dataslices
    }

    torch.save(embs_datapool, os.path.join(save_path, 'dreamsim_ensemble_emb_p128_'+dataset_name+'.pt'))

def cli_main(args):
    device = args.device
    save_path = args.save_path
    dataset_json = args.dataset_json
    model, preprocess = dreamsim(pretrained=True, device=device, dreamsim_type="ensemble")
    compute_embeddings(save_path, dataset_json, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse datasets.")
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=True,
        help="path for saving the embeddings",
    )
    parser.add_argument(
        "-d",
        "--dataset_json",
        type=str,
        required=True,
        help="dataset json file to filter",
    )
    parser.add_argument(
        "-c",
        "--device",
        type=str,
        default='cuda:0',
        help="device to use for processing",
    )

    
    args = parser.parse_args()
    cli_main(args)


