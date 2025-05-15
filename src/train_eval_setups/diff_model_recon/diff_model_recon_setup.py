from src.interfaces.base_train_eval_setup import BaseTrainEvalSetup
from src.interfaces.config_models import *
from src.train_eval_setups.train_eval_setups import register_train_eval_setup

from .utils.wandb_utils import wandb_kwargs_via_cfg

import hydra
import yaml

from omegaconf import OmegaConf

from .train_diff_models import coordinator as train_diff_models_coord
#from .reconstruct import coordinator as reconstruct_coord

from pathlib import Path
import os
import sys

from glob import glob
from .utils.utils import natural_sort
from dataclasses import dataclass
import logging

from src.interfaces.path_setups import *

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as



class DiffModelVariationSetupConfig(BaseModel):
    exp_train          : str    # Link to the hydra exp, which defines the concrete training setup (e.g. mri2d_multicoil/train)
    exp_eval           : str    # Link to the hydra exp, which defines the concrete evaluation setup (e.g. mri2d_multicoil/variational)
    hydra_config_path  : str    # Base path to the hydra config.yaml file
    hydra_config_name  : str    # Name of the hydra config.yaml file
    eval_use_ema_model : bool    # Whether to look for ema models in the training output dir for choosing the evaluation model
    eval_model_glob    : str    # A glob identifier to look for models (can be used to choose a concrete checkpoint)
    # Optional arguments (for wandb logging, maybe remove later)
    train_descr        : str    #
    train_note         : str    # 
    eval_descr         : str    # 
    eval_note          : str    # 

@register_train_eval_setup()
class DiffModelVariationSetup(BaseTrainEvalSetup):

    def __init__(self):
        super().__init__(self.__class__.__name__)

    def _initialize_setup(self, setup_config_path : str, setup_overwrites : str):
        # parse setup config
        self.setup_config = parse_yaml_file_as(DiffModelVariationSetupConfig, setup_config_path)
        if setup_overwrites == "":
            self.setup_overwrites = []
        else:
            self.setup_overwrites = setup_overwrites.split(" ")

    def _run_train(self, train_dataset_config : TrainDatasetConfigModel, train_dataset_base_path : str):
        """
            Runs the training code specific to the setup.
            -> Save logs and checkpoints *in the current working directory*.
        """
        from hydra import initialize, compose
        with initialize(version_base=None, config_path=self.setup_config.hydra_config_path):
            cfg = compose(config_name=self.setup_config.hydra_config_name, overrides=[
                f"++hydra.run.dir=.",
                f"+exps={self.setup_config.exp_train}",
                f"++wandb.descr_short={self.setup_config.train_descr}",
                f"++wandb.note={self.setup_config.train_note}"
            ] + self.setup_overwrites)
            train_diff_models_coord(cfg, train_dataset_config, train_dataset_config_paths_base=train_dataset_base_path, resume=self.resume)

    def _get_chosen_model_checkpoint_path_for_eval(self, train_output_dir : str) -> str:
        """
            Decide on the chosen model checkpoint for evaluation.
            -> Setups 
        """
        path = os.path.join(train_output_dir, self.setup_config.eval_model_glob)
        sorted_paths = natural_sort(glob(path))
        assert len(sorted_paths) > 0, f"No model checkpoint found in {path} with glob {self.setup_config.eval_model_glob}"
        return sorted_paths[-1] # take last model (with hightest epoch nr)
    

    def _run_eval_on_model_checkpoint(self, eval_dataset_config : EvalDatasetConfigModel, eval_dataset_base_path : str, model_checkpoint : str) -> EvalOutputModel:
        """
            Runs the evaluation code specific to the setup.
            -> Save logs and reconstruction results *in the current working directory*.
            -> Return the evaluation output model.
        """
        from hydra import initialize, compose
        from .evaluate import main as evaluate_main
        with initialize(version_base=None, config_path=self.setup_config.hydra_config_path):
            ret_json = None
            cfg = compose(config_name=self.setup_config.hydra_config_name, overrides=[
                f"++hydra.run.dir=.",
                f"+exps={self.setup_config.exp_eval}",
                f"++wandb.descr_short={self.setup_config.eval_descr}",
                f"++wandb.note={self.setup_config.eval_note}"] + self.setup_overwrites)

            if self.setup_config.eval_use_ema_model:
                cfg.reconstruct.load_ema_params_from_path = model_checkpoint
            else:
                cfg.reconstruct.load_params_from_path = model_checkpoint

            # setup config model_path is passed since it is saved in the results json
            ret_json = evaluate_main(cfg, eval_dataset_config=eval_dataset_config, eval_dataset_config_paths_base=eval_dataset_base_path)

            return EvalOutputModel.model_validate_json(ret_json)