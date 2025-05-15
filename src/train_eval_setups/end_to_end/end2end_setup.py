from src.interfaces.base_train_eval_setup import BaseTrainEvalSetup
from src.interfaces.config_models import *

from src.train_eval_setups.train_eval_setups import register_train_eval_setup

import os
import yaml
from src.train_eval_setups.end_to_end.utils import natural_sort
import json
from pathlib import Path
import logging

from src.interfaces.path_setups import *

from src.train_eval_setups.end_to_end import train, evaluate

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as

from dataclasses import dataclass

# TODO: split compute-specific aspects from the setup-specific aspects
@dataclass
class Args():
    model_setup : str = None
    train_setup : str = None
    eval_setup : str = None
    output_dir : str = None
    outfile : str = None
    finetune : str = None
    num_checkpoints : int = 1
    num_workers : int = 4
    resume : bool = False
    world_size : int = 1
    model_seed : int = 0

@register_train_eval_setup()
class End2EndSetup(BaseTrainEvalSetup):

    def __init__(self):
        super().__init__(self.__class__.__name__)

    def _initialize_setup(self, setup_config_path : str, setup_overwrites : str):
        """
            Initialize the method, such as parsing the setup-specific configuration files, and potentially loading some libaries.

            Parameters:
                - setup_config_path : str : path to the setup configuration file.

            Returns: nothing
        """
        with open(setup_config_path, "r") as f:
            self.model_config = yaml.safe_load(f)
        self.setup_config_path = setup_config_path # is currently needed in eval for logging reasons
        
        self.args = Args()

        # # path setup
        # self.args.output_dir, self.outfile_eval = setup_train_eval_paths(
            # base_path_config, self, setup_config_path, train_dataset_config, eval_dataset_config)

        # self.outfile_model = os.path.join(self.args.output_dir, 'model.json')
        self.args.dir_to_exp_data = os.path.join(Path(os.path.realpath(__file__)).parents[3], 'exp_data')        

        if setup_overwrites != "":
            logging.warning(f"Overwrites are not supported for this setup (provided: {setup_overwrites}).")

    def _run_train(self, train_dataset_config : TrainDatasetConfigModel, train_dataset_base_path : str):
        """
            Runs the training code specific to the setup.

            Note: Save logs and checkpoints *in the current working directory*.

            Parameters:
                - train_dataset_config : TrainDatasetConfigModel : the training dataset configuration
                - train_dataset_base_path : str : the base path to the training dataset (prepend to each dataset in the TrainDatasetConfigModels)

            Returns: nothing
        """
        self.args.output_dir = os.getcwd()
        exp_name = Path(self.args.output_dir).stem
        self.args.path_to_model_summary = os.path.join(self.args.dir_to_exp_data, 'models', exp_name + '.json')
        self.args.resume = self.resume
        self.args.finetune = self.finetune
        train.main(self.args, model_config=self.model_config, train_config=train_dataset_config, train_dataset_base_path=train_dataset_base_path)

    def _get_chosen_model_checkpoint_path_for_eval(self, train_output_dir_path : str) -> str:
        """
            Return the full path to the chosen training checkpoint.

            Which checkpoint is loaded is specified in the setup-specific configuration files.

            Parameters:
                - train_output_dir_path : str : the path to the training output directory

            Returns:
                - str : the full path to the chosen model checkpoint
        """
        exp_name = Path(train_output_dir_path).stem
        path_to_model_summary = os.path.join(self.args.dir_to_exp_data, 'models', exp_name + '.json')
        with open(path_to_model_summary) as f:
            model_summary = json.load(f)
            checkpoints = model_summary['checkpoints']
        # for name in natural_sort(checkpoints.keys())[::-1]:
            # model_path = checkpoints[name] 
        
        # return model_path
        self.args.eval_checkpoint = natural_sort(checkpoints.values())[-1]
        return self.args.eval_checkpoint

    def _run_eval_on_model_checkpoint(self, eval_dataset_config : EvalDatasetConfigModel, eval_dataset_base_path : str, model_checkpoint : str) -> EvalOutputModel:
        """
            Runs the evaluation code specific to the setup.

            Note: Save logs and reconstruction results *in the current working directory*.

            Parameters:
                - eval_dataset_config : EvalDatasetConfigModel : the evaluation dataset configuration
                - eval_dataset_base_path : str : the base path to the evaluation dataset (prepend to each dataset in the EvalDatasetConfigModels)
                - model_checkpoint : str : the full path to the model checkpoint to evaluate

            Returns:
                - EvalOutputModel : the evaluation output model (the summary of the evaluation)
        """
        self.args.output_dir = os.getcwd()
        exp_name = Path(self.args.output_dir).stem
        self.args.outfile = os.path.join(self.args.dir_to_exp_data, 'evals', exp_name + '_' + Path(self.args.eval_checkpoint).stem + '.json')
        self.args.model_path = model_checkpoint

        ret_json = evaluate.main(self.args, setup_config=self.model_config, setup_config_file=self.setup_config_path, eval_config=eval_dataset_config, eval_dataset_base_path=eval_dataset_base_path)
        return EvalOutputModel.model_validate_json(ret_json) # take last result, validate its format, and return the python object