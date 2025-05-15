from typing import Any, Dict, Optional

import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import wandb
import os

from src.train_eval_setups.diff_model_recon.utils.wandb_utils import tensor_to_wandbimages_dict

import logging

from .loss import epsilon_based_loss_fn
from .sde import SDE
from .utils import save_model
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from .ddim_sampler import DDIM

from src.train_eval_setups.diff_model_recon.physics_trafos.trafos_prior_target.BasePriorTrafo import BasePriorTrafo
from src.train_eval_setups.diff_model_recon.diffmodels.networks import UNetModel, ExponentialMovingAverage
from src.interfaces.config_models import TrainDatasetConfigModel
from src.train_eval_setups.diff_model_recon.utils.utils import get_dataloader
from ...common_utils.datasets.dataset_factory import get_dataset

from src.train_eval_setups.diff_model_recon.utils.ddp_utils import cache_iterable_in_memory

from ..utils import natural_sort
from glob import glob
import warnings

def wandb_stats(wandb_run, dataloader, epoch, batch_size, log_samples_nr=2):
    samples_mean = torch.zeros(len(dataloader) * batch_size)
    samples_std = torch.zeros(len(dataloader) * batch_size)
    samples_norm = torch.zeros(len(dataloader) * batch_size)

    with tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
        for i, x in pbar:
            if i < log_samples_nr:
                wandb_run.log({
                    'global_step': i,
                    'step' : i,
                    **tensor_to_wandbimages_dict(f"data_samples_{i}", x[0] if x[0].ndim >= 3 else x[0].unsqueeze(-1), show_phase=False, take_meanslices=True) # if only two dims (RSS images), then add another dim at the end
                })
            if x.shape[0] != batch_size:
                continue
            x = x.view(batch_size,-1)
            samples_mean[i*batch_size:(i+1)*batch_size] = x.mean(dim=-1)
            samples_std[i*batch_size:(i+1)*batch_size] = x.std(dim=-1)
            samples_norm[i*batch_size:(i+1)*batch_size] = torch.linalg.norm(x, dim=-1)

    wandb_run.log({
        'samples_count': len(dataloader),
        'sample_mean_mean': samples_mean.mean(),
        'sample_mean_std': samples_mean.std(),
        'sample_std_mean': samples_std.mean(),
        'sample_std_std': samples_std.std(),
        'sample_norm_mean': samples_norm.mean(),
        'sample_norm_std': samples_norm.std(),
        'global_step' : epoch,
        'step' : epoch
    })


def score_model_trainer(
    score: UNetModel,
    sde: SDE,
    optim_kwargs: Dict,
    val_kwargs: Dict,
    prior_trafo: BasePriorTrafo,
    use_ema: bool = True,
    device: Optional[Any] = None, 
    sample_each_nth_epoch: Optional[int] = None, 
    save_img_each_sample_iteration: Optional[int] = None,
    train_dataset_config :  TrainDatasetConfigModel = None,
    train_dataset_config_paths_base : str = "",
    dataset_trafo : Optional[Any] = None,
    resume: bool = False
    ):
    

    optimizer = Adam(score.parameters(), lr=0)

    loss_fn = epsilon_based_loss_fn 

    ema = None
    if use_ema: 
        ema = ExponentialMovingAverage(
            score.parameters(),
            decay=optim_kwargs['ema_decay']
            )

    start_epoch = 0
    if resume:
        if len(glob('./model_*[0-9].pt')) > 0:
            score_cp = natural_sort(glob('./model_*[0-9].pt'))[-1]
            ema_cp = natural_sort(glob('./ema_model_*[0-9].pt'))[-1]
            assert score_cp.split('_')[-1] == ema_cp.split('_')[-1], f'ema checkpoint not the same epoch as score checkpoint {ema_cp}, {score_cp}'
            score.load_state_dict(torch.load(score_cp, map_location=device))
            print("Score model loaded.")
            logging.info(f'Score model ckpt loaded from: {score_cp}')
            ema = ExponentialMovingAverage(
                score.parameters(),
                decay=optim_kwargs['ema_decay']
                )
            ema.load_state_dict(torch.load(ema_cp, map_location=device))
            print("EMA model loaded.")
            logging.info(f'EMA model ckpt loaded from: {ema_cp}')
            start_epoch = int(score_cp.split('_')[-1].split('.')[0]) + 1
            print(f"Resume training at epoch {start_epoch}")
        else:
            pass
    batch_size = optim_kwargs['batch_size']

    dataset_path_previous = None
    use_preloaded_dataset = True

    grad_step = 0
    cache_in_gpu = False
    max_epochs = len(train_dataset_config.epochs_setup)
    for epoch in range(start_epoch, max_epochs):

        # here we have to create the correct dataset
        epochn = epoch + 1
        if not epochn in train_dataset_config.epochs_setup:
            logging.warning(f"Epoch {epochn} not in train_dataset_config.epochs_setup. Stopping training.")
            break
        dataset_name = train_dataset_config.epochs_setup[epochn]

        # setup for single-gpu training
        world_size = 1
        rank = 0

        num_workers = optim_kwargs['dataloader_num_workers']
        augment_data = False

        dataset_path = os.path.join(train_dataset_config_paths_base, dataset_name)
        dataset_as_in_previous_epoch = (use_preloaded_dataset and dataset_path_previous == dataset_path)
        dataset_path_previous = dataset_path

        if not dataset_as_in_previous_epoch:

            if optim_kwargs["cache_dataset"] and optim_kwargs["cache_dataset_load_from_disk"]:
                #path = cfg.diffmodels.train.cache_dataset_disk_path
                dataset_name_stem = Path(dataset_name).stem
                path = os.path.join(optim_kwargs["cache_dataset_disk_path"], dataset_name_stem + ".pt")
                # try:
                #     dataset = torch.load(path, map_location=device)
                #     logging.warn(f"Loaded dataset from: {path}, use for debugging only!")
                # except:
                dataset = get_dataset(dataset_path, transform=dataset_trafo, augment_data=augment_data)
                logging.warning(f"Could not load dataset from disk at path: {path}. Creating new dataset.")
            else:
                dataset = get_dataset(dataset_path, transform=dataset_trafo, augment_data=augment_data) 

            # cache dataset if enabled: useful for debugging
            if optim_kwargs["cache_dataset"]:
                cache_in_gpu = optim_kwargs["cache_dataset_in_gpu"]
                cache_device = device if cache_in_gpu  else "cpu"
                dataset = cache_iterable_in_memory(
                    iterable_ds=dataset, use_tqdm=True, device=cache_device, repeat_dataset=optim_kwargs["cache_dataset_repeats"]
                    )
                if optim_kwargs["cache_dataset_store_on_disk"]:
                    dataset_name_stem = Path(dataset_name).stem
                    path = os.path.join(optim_kwargs["cache_dataset_disk_path"], dataset_name_stem + ".pt")
                    torch.save(dataset, path)

        # here it would be good to cache the dataset in the GPU if possible
        dataloader = get_dataloader(dataset,
            num_workers=num_workers if not cache_in_gpu else 0,
            batch_size=batch_size, rank=rank, world_size=world_size, group_shape_by="target")
        if world_size > 1:
            dataloader.batch_sampler.set_epoch(epoch)
        if epoch == 0:
            scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, 
            max_lr=optim_kwargs['lr'],
            total_steps=max_epochs*len(dataloader),
            pct_start=0.01,
            anneal_strategy='linear',
            cycle_momentum=False,
            base_momentum=0., 
            max_momentum=0.,
            div_factor = 25,
            final_div_factor=100,
            )
        # # # calculate stats
        # if wandb.run is not None and not dataset_as_in_previous_epoch:
        #     wandb_stats(wandb.run, dataloader, batch_size=batch_size, epoch=epoch,
        #         log_samples_nr=optim_kwargs["log_samples_nr"])

        im_shape = prior_trafo(next(iter(dataloader))).shape[-3:]
        avg_loss, num_items = 0, 0
        score.train()
        with tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
            for _, x in pbar:
                x = prior_trafo(x.to(device))
                ## mvue alrady has a channel dim for the complex numbers (real, imag)
                #x = x.squeeze(1)
                loss = loss_fn(
                    x=x,
                    model=score,
                    sde=sde
                    )

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(score.parameters(), max_norm=1, norm_type=2.)
                optimizer.step()
                try:
                    scheduler.step()
                except ValueError:
                    warnings.warn("Scheduler total steps reached.")
                pbar.set_description(
                    f'loss={loss.item():.1f}',
                    refresh=False
                )
                
                grad_step += 1
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]

            should_save_model = (
                epoch % optim_kwargs['save_model_every_n_epoch'] == 0 or epoch == max_epochs - 1)
            
            if use_ema and (
                grad_step > optim_kwargs['ema_warm_start_steps'] or epoch > 0):
                ema.update(score.parameters())
            
            if should_save_model:
                save_model(score=score, epoch=epoch, max_epochs=max_epochs, ema=ema)
            

            print(f'epoch: {epoch}, loss: {avg_loss / num_items}')
            # wandb.log(
            #     {'loss': avg_loss / num_items, 'epoch': epoch + 1, 'step': epoch + 1}
            # )

            # if sample_each_nth_epoch is not None:
            #     if epoch % sample_each_nth_epoch == 0:
            #         if use_ema:
            #             ema.store(score.parameters())
            #             ema.copy_to(score.parameters())
            #             score = score.to(device)
            #         score.eval()
            #         sampler = DDIM(
            #         score=score,
            #         sde=sde,
            #         sample_kwargs={
            #             'num_steps': val_kwargs['num_steps'],
            #             'batch_size': val_kwargs['batch_size'],
            #             'eps': val_kwargs['eps'],
            #             'eta': val_kwargs['eta'],
            #             'im_shape': im_shape
            #             },
            #         save_img_each_sample_iteration = save_img_each_sample_iteration,
            #         device=device
            #         )
            #         sample = sampler.sample()

            #         if wandb.run is not None:
            #             wandb.run.log({
            #                 'sample_mean': sample.mean(),
            #                 'sample_std': sample.std()
            #                 })

            #         if use_ema: ema.restore(score.parameters())

    torch.save(score.state_dict(), 'last_model.pt')
