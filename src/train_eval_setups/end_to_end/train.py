import argparse
import yaml
import os
from glob import glob
from tqdm.autonotebook import tqdm
import numpy as np

from fastmri.losses import SSIMLoss
from fastmri.data.transforms import center_crop

from tqdm.autonotebook import tqdm
from runstats import Statistics

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils import logging, natural_sort
from .utils import get_dataloader, setup

import warnings
import uuid
from datetime import date
import json
from pathlib import Path
from glob import glob

def train(
    rank,
    master_port,
    world_size,
    model_config,
    train_config,
    train_dataset_base_path,
    output_dir,
    finetune,
    num_workers,
    num_checkpoints,
    resume,
    model_seed,
):  
    accel_factors = model_config['accel_factors']
    lr = model_config['lr']
    batch_size = model_config['batch_size']
    #num_epochs = train_config['num_epochs']
    #samples_seen = train_config['samples_seen']
    num_epochs = train_config.num_epochs
    samples_seen = train_config.samples_seen
    total_steps = samples_seen//batch_size
    augment_data = False
    assert batch_size % world_size == 0, 'batch_size must be divisible by world_size'

    device = 'cuda:%d' % rank 
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = master_port

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        batch_size = batch_size//world_size
        
    model, data_transform, forward_fn = setup('train', model_config, accel_factors, seed=model_seed)
    model = model.to(device)

    if finetune is not None:
        checkpoint = torch.load(finetune, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    loss_fn = SSIMLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.01,
        anneal_strategy='linear',
        cycle_momentum=False,
        base_momentum=0., 
        max_momentum=0.,
        div_factor = 25.,
        final_div_factor=1.,
    )

    # Training
    if not resume:
        logging(output_dir, 'Start training')
        first_epoch = 1
    else:
        assert glob(os.path.join(output_dir, 'checkpoints/*.pt')), 'Resume training failed: No checkpoints saved.'
        logging(output_dir, 'Resume training.')
        checkpoint = natural_sort(glob(os.path.join(output_dir, 'checkpoints/*.pt')))[-1]
        checkpoint = torch.load(checkpoint, map_location=device)
        checkpoint_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        logging(output_dir, 'Model from epoch {:d} loaded.'.format(checkpoint_epoch))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging(output_dir, 'Optimizer loaded. Current learning rate: {:f}'.format(optimizer.param_groups[0]['lr']))
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging(output_dir, 'Scheduler loaded. Current learning rate: {:f}'.format(scheduler.get_last_lr()[0]))
        first_epoch = checkpoint_epoch+1
        assert first_epoch <= num_epochs, 'Cannot resume. Training is already finished.'
        num_epochs = num_epochs-checkpoint_epoch
        print('Resume training at epoch {:d}'.format(first_epoch))
        logging(output_dir, 'Resume training at epoch {:d}'.format(first_epoch))

    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    model.train()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    path_to_checkpoints = os.path.join(output_dir, 'checkpoints')
    if rank == 0:
        if not os.path.exists(path_to_checkpoints):
            os.makedirs(path_to_checkpoints)
            print(f"Created {path_to_checkpoints}")
        else:
            print(f"{path_to_checkpoints} already exist. Continue.")
        
    for epoch in tqdm(range(first_epoch, first_epoch+num_epochs)):
        dataset = train_config.epochs_setup[epoch]
        dataset_path = os.path.join(train_dataset_base_path, dataset)
        dataloader = get_dataloader(dataset_path, data_transform, num_workers, augment_data, batch_size=batch_size, rank=rank, world_size=world_size, load_sensitivity_maps=False)

        if world_size > 1:
            dataloader.batch_sampler.set_epoch(epoch)

        total_loss = Statistics()

        for sample in tqdm(dataloader):
            optimizer.zero_grad(set_to_none=True)
            target = sample.target.unsqueeze(-3).to(device)
            maxval = sample.max_value.to(device)
            output = forward_fn(model, sample, device)
            output = center_crop(output, target.shape[-2:])
            loss = loss_fn(output, target, maxval)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2.)
            optimizer.step()
            try:
                scheduler.step()
            except ValueError:
                warnings.warn("Scheduler total steps reached.")
            total_loss.push(loss.item())

        train_loss = total_loss.mean()

        # if world_size > 1:
            # dist.barrier()

        if rank == 0:
            with open(os.path.join(output_dir,'train_loss.txt'), 'a') as f:
                f.write(str(train_loss)+'\n')
            if epoch % num_checkpoints == 0:
                checkpoint = { 
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()}
                torch.save(checkpoint, os.path.join(path_to_checkpoints, 'epoch_'+str(epoch)+'.pt'))
            print('Epoch {}, Train loss.: {:0.4f}'.format(epoch, train_loss))

    logging(output_dir, 'Finish training')

    if world_size > 1:
        print('Destroying process group...')
        dist.destroy_process_group()
        print('Done.')

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_setup", 
        type=str,
        required=True,
        help="Configuration of the model setup as yaml file (model, acceleration factor, batch_size, learning_rate)"
    )
    parser.add_argument(
        "--train_setup", 
        type=str,
        required=True,
        help="Configuration of the training setup as yaml file (which dataset at which epoch)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        required=True,
        help="Path to directory where checkpoints and logs will be stored."
    )
    parser.add_argument(
        "--path_to_model_summary", 
        type=str,
        default = None,
        help="Name and path to output file as .json"
    )
    parser.add_argument(
        "--finetune", 
        type=str, 
        default=None,
        help="Path to checkpoint for finetuning"
    )
    parser.add_argument(
        "--num_checkpoints", 
        type=int, 
        default=5, # before: 5, but should be part of configuration anyways
        help="Saves a checkpoint after each 'num_checkpoints' epochs of training."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of workers for pytorch dataloader"
    )
    parser.add_argument(
        "--resume", 
        action='store_true',
        default=False,
        help="Whether to resume training from last checkpoints contained in --output_dir"
    )
    parser.add_argument(
        "--world_size", 
        type=int, 
        default=1,
        help="Set larger than 1 for DDP."
    )
    parser.add_argument(
        "--model_seed", 
        type=int, 
        default=0,
        help="Random seed for init the model"
    )
    
    return parser
    

def main(args, model_config, train_config, train_dataset_base_path):
 
    train_args = {
        'world_size': args.world_size,
        'model_config': model_config,
        'train_config': train_config,
        'train_dataset_base_path' : train_dataset_base_path,
        'output_dir': args.output_dir,
        'finetune': args.finetune,
        'num_workers': args.num_workers,
        'num_checkpoints': args.num_checkpoints,
        'resume': args.resume,
        'model_seed': args.model_seed,
    }

    if train_args['world_size'] > 1:
        master_port = str(np.random.randint(15000,65535)); print(f"master port: {master_port}")
        mp.spawn(train, args=(master_port, )+tuple(train_args.values()), nprocs=train_args['world_size'], join=True)
    else:
        train(0, None, **train_args)

    exp_setup = dict({k:v for k,v in train_args.items() if k not in ['output_dir', 'resume', 'train_config']})
    exp_setup["train_config"] = train_config.dict()

    model_data = {
        "name": train_config.name,
        "uuid": str(uuid.uuid4()),
        "creation_date": str(date.today()),
        "output_dir": os.getcwd(),
        "checkpoints": dict((Path(f).stem, f) for f in natural_sort(glob(os.path.join(train_args['output_dir'], 'checkpoints', '*.pt')))),
        "exp_setup": exp_setup
    }
    model_data_json = json.dumps(model_data, indent=4)
    with open(args.path_to_model_summary, 'w') as outfile:
        outfile.write(model_data_json)

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    main(args)
    

