import hydra
import torch

import wandb

from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from .diffmodels.networks import load_score_model
from .diffmodels import score_model_trainer, load_sde_model
from .utils.wandb_utils import wandb_kwargs_via_cfg

from src.interfaces.config_models import TrainDatasetConfigModel

from .physics_trafos.trafo_utils import get_standard_prior_trafo, get_standard_dataset_transform
import logging

def coordinator(cfg : DictConfig,
    train_dataset_config : TrainDatasetConfigModel,
    train_dataset_config_paths_base : str,
    resume: bool,
    ) -> None:

    OmegaConf.resolve(cfg)

    # store yaml config in current directory
    with open('hydra_config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    wandb_kwargs = wandb_kwargs_via_cfg(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #logging.getLogger().setLevel(logging.INFO)
    logging.info("Using device: %s", device)

    with wandb.init(**wandb_kwargs) as run:

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        transform_device = "cpu" if cfg.diffmodels.train.dataloader_num_workers > 0 and not cfg.diffmodels.train.cache_dataset_in_gpu else device
        dataset_trafo = get_standard_dataset_transform(cfg,
            device=transform_device,
            provide_pseudinverse=False,
            provide_measurement=False
        )
        sde = load_sde_model(cfg)
        score = load_score_model(cfg, device=device)

        prior_trafo = get_standard_prior_trafo(cfg)

        wandb.run.summary['num_params_score'] = sum(p.numel() for p in score.parameters() if p.requires_grad)

        optim_kwargs = dict(cfg.diffmodels.train)
        val_kwargs = dict(cfg.diffmodels.val.sampling)
        #val_kwargs['im_shape'] = (1, *dataset.shape)

        kwargs = {'sample_each_nth_epoch': cfg.diffmodels.val.sample_each_nth_epoch, 
            'save_img_each_sample_iteration': cfg.diffmodels.val.save_img_each_sample_iteration,
            'resume': resume
            }

        score_model_trainer(    
            score=score,
            sde=sde,
            use_ema=cfg.diffmodels.train.use_ema,
            optim_kwargs=optim_kwargs,
            val_kwargs=val_kwargs,
            prior_trafo=prior_trafo,
            device=device, 
            dataset_trafo=dataset_trafo,
            train_dataset_config=train_dataset_config,
            train_dataset_config_paths_base=train_dataset_config_paths_base,
            **kwargs
        )

        torch.save(score.state_dict(), 'last_model.pt')

if __name__ == '__main__':
    coordinator()  