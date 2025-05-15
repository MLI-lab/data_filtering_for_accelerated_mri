import os
from dreamsim import dreamsim
import torch
import json
from tqdm import tqdm
from torchmetrics.functional import pairwise_cosine_similarity
import uuid
import argparse


def cli_main(args):
    device = args.device
    dataset_json = args.dataset_json_filter
    emb_train = torch.load(args.path_to_embeddings_filter)
    emb_test = torch.load(args.path_to_embeddings_ref)
    knn = args.knn
    model, preprocess = dreamsim(pretrained=True, device=device, dreamsim_type="ensemble")

    fnames_list = list(set(emb_train['fnames']))
    fnames_dict = {}
    for i in range(len(fnames_list)):
        fnames_dict[fnames_list[i]] = i
        
    for i in range(len(emb_train['fnames'])):
        emb_train['fnames'][i] = fnames_dict[emb_train['fnames'][i]]

    zero = torch.zeros(1,3,128,128)
    zero_emb = model.embed(zero.to(model.device)).to(torch.float16).to(device)
    zero_th = 0.6


    embs = emb_train['embs'].to(device)
    fnames = torch.Tensor(emb_train['fnames']).to(device).to(torch.int)
    slices = emb_train['slices'].to(device).to(torch.int)

    sim = pairwise_cosine_similarity(embs, zero_emb).squeeze()
    embs = embs[sim<zero_th]
    fnames = fnames[sim<zero_th]
    slices = slices[sim<zero_th]

    emb_train['embs'] = embs
    emb_train['fnames'] = fnames
    emb_train['slices'] = slices

    e_test = emb_test['embs'].to(device)
    sim = pairwise_cosine_similarity(e_test, zero_emb).squeeze()
    e_test = e_test[sim<zero_th]

    by_volume = {}
    for i, s, e in tqdm(zip(emb_train['fnames'], emb_train['slices'], emb_train['embs'])):
        i = i.item()
        if i not in by_volume.keys():
            by_volume[i] = [[], [], []]
        by_volume[i][0].append(s.item())
        by_volume[i][1].append(e)

    for k, v in tqdm(by_volume.items()):
        slices = torch.tensor(v[0])
        sorted_slices, sorted_idx = torch.sort(slices)
        sorted_embs = torch.stack(v[1])[sorted_idx]
        by_volume[k] = [sorted_slices, sorted_embs, []]

    for k, v in tqdm(by_volume.items()):
        keep = torch.tensor([True] * len(v[1]))
        for i, emb in enumerate(v[1]):
            if keep[i]:
                sims = pairwise_cosine_similarity(emb.view(1,-1), v[1][i+1:]).squeeze()
                keep[i+1:][sims>9e-1] = False # dedup
        by_volume[k][2] = keep

    c = 0
    q = []
    fnames_dedup = []
    slices_dedup = []
    embs_dedup = []
    for k, v in tqdm(by_volume.items()):
        slices = v[0]
        embs = v[1]
        keep = v[2]
        for s, emb in zip(slices[keep], embs[keep]):
            fnames_dedup.append(k)
            slices_dedup.append(s)
            embs_dedup.append(emb)
    embs_dedup = torch.stack(embs_dedup)
    fnames = torch.tensor(fnames_dedup)
    slices = torch.tensor(slices_dedup)

    sims = pairwise_cosine_similarity(e_test.to('cpu'), embs_dedup.to('cpu'))

    filtered = []
    for sim in tqdm(sims):
        idx = torch.argsort(sim.to(device), descending=True)[:knn].cpu()
        a = torch.unique(torch.stack((fnames[idx], slices[idx])), dim=1, sorted=False)
        filtered.append(a)
    filtered = torch.unique(torch.concatenate(filtered, dim=1), dim=1, sorted=False)
    fnames_filtered = [fnames_list[i] for i in filtered[0,:]]
    slices_filtered = [i.item() for i in filtered[1,:]]
    final_filtered = list(zip(fnames_filtered, slices_filtered))
    
    num_slices = len(final_filtered)
    subset = final_filtered
    dataset = {}

    filter_fname = args.path_to_embeddings_filter.split('/')[-1].split('.')[0]
    slice_subset = {
        "dataset_name": f'{filter_fname}',
        "uuid": str(uuid.uuid4()),
        "num_slices": num_slices,
        "files": {}
    }
    with open(dataset_json) as f:
        json_data = json.load(f)
    for f, s in subset:
        s = str(s)
        if f not in slice_subset['files']:
            slice_subset['files'][f] = json_data['files'][f].copy()
            slice_subset['files'][f]['slices'] = {}
        slice_subset['files'][f]['slices'][s] = json_data['files'][f]['slices'][s]

    data_json = json.dumps(slice_subset, indent=4)

    with open(os.path.join('datasets/train/', slice_subset['dataset_name'] + '.json'), 'w') as outfile:
        outfile.write(data_json)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval filter dataset.")
    parser.add_argument(
        "-c",
        "--device",
        type=str,
        default='cuda:0',
        help="Device to use for computation.",
    )
    parser.add_argument(
        "-d",
        "--dataset_json_filter",
        type=str,
        required=True,
        help="Dataset json file to filter.",
    )
    parser.add_argument(
        "-f",
        "--path_to_embeddings_filter",
        type=str,
        required=True,
        help="Path to the embeddings to filter.",
    )
    parser.add_argument(
        "-r",
        "--path_to_embeddings_ref",
        type=str,
        required=True,
        help="Path to the reference embeddings.",
    )
    parser.add_argument(
        "-k",
        "--knn",
        type=int,
        required=True,
        help="Number of nearest neighbors to consider.",
    )
    
    args = parser.parse_args()
    cli_main(args)


