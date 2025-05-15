# %%
from abc import ABC, abstractmethod
from typing import *

import logging
import os
from pathlib import Path
from src.interfaces.config_models import EvalDatasetConfigModel, TrainDatasetConfigModel, BasePathConfigModel, EvalOutputModel
from src.interfaces.path_setups import setup_train_output_dir_path, setup_eval_output_dir_path, setup_eval_summary_file_path, import_bart_from_path_model

class BaseTrainEvalSetup(ABC):
    def __init__(self, name):
        self.name = name

    ######## Methods which need to be implemented by the specific setup ########
    @abstractmethod
    def _initialize_setup(self, setup_config_path : str, setup_overwrites : str):
        """
            Initialize the method, such as parsing the setup-specific configuration files, and potentially loading some libaries.

            Parameters:
                - setup_config_path : str : path to the setup configuration file.

            Returns: nothing
        """
        pass

    @abstractmethod
    def _run_train(self, train_dataset_config : TrainDatasetConfigModel, train_dataset_base_path : str):
        """
            Runs the training code specific to the setup.

            Note: Save logs and checkpoints *in the current working directory*.

            Parameters:
                - train_dataset_config : TrainDatasetConfigModel : the training dataset configuration
                - train_dataset_base_path : str : the base path to the training dataset (prepend to each dataset in the TrainDatasetConfigModels)

            Returns: nothing
        """
        pass

    @abstractmethod
    def _get_chosen_model_checkpoint_path_for_eval(train_output_dir_path : str) -> str:
        """
            Return the full path to the chosen training checkpoint.

            Which checkpoint is loaded is specified in the setup-specific configuration files.

            Parameters:
                - train_output_dir_path : str : the path to the training output directory

            Returns:
                - str : the full path to the chosen model checkpoint
        """
        pass

    @abstractmethod
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
        pass

    ######## Common behavior and interface to all setups ########
    def initialize(self,
        base_path_config : BasePathConfigModel, 
        setup_config_path : str,
        setup_overwrites : str,
        train_dataset_config : TrainDatasetConfigModel,
        eval_dataset_config : EvalDatasetConfigModel,
        resume: bool,
        finetune: str,
        ):
        self.finetune = finetune
        self.resume = resume
        self.__base_path_config = base_path_config
        self.__setup_config_path = setup_config_path
        self.__train_dataset_config = train_dataset_config
        self.__eval_dataset_config = eval_dataset_config

        # path setup (for training and evaluation outputs)
        self.__train_output_dir_path = setup_train_output_dir_path(base_path_config, self.name, setup_config_path, train_dataset_config)
        self.__eval_output_dir_path = setup_eval_output_dir_path(base_path_config, self.name, setup_config_path, train_dataset_config, eval_dataset_config)

        # by convention we save the loaded model checkpoint in the summary file name
        logging.info(f"train_output_dir_path: {self.__train_output_dir_path}")
        logging.info(f"eval_output_dir_path: {self.__eval_output_dir_path}")
        # the eval summary file then depends on the checkpoint, which is known after training

        # import the bart-specific libs (which is used by multiple setups)
        import_bart_from_path_model(base_path_config)
        
        # finally we run setup-specific initialization code
        self._initialize_setup(setup_config_path, setup_overwrites)

    def train(self):
        """
            Runs the training stage of the setup.
            Saves training results to the training output directory only.
        """
        curdir = os.getcwd()
        os.chdir(self.__train_output_dir_path)
        try:
            # copy the setup config to the current dir
            os.system(f"cp {self.__setup_config_path} ./{Path(self.__setup_config_path).name}")
            self._run_train(self.__train_dataset_config, train_dataset_base_path=curdir)
        finally:
            os.chdir(curdir)

    def _get_summary_file_path(self):
        checkpoint_name = Path(self._get_chosen_model_checkpoint_path_for_eval(self.__train_output_dir_path)).stem
        return setup_eval_summary_file_path(self.__base_path_config, self.name, self.__setup_config_path, self.__train_dataset_config, self.__eval_dataset_config, extra_info=checkpoint_name, file_ext=".json")

    def eval(self) -> EvalOutputModel:
        """
            Runs the evaluation stage of the setup.
            Stores the following data:
                - evaluation results in the evaluation output directory (reconstructions, logs etc)
                - a separte summary file to the eval output summary path
            Returns the validated evaluation results.
        """
        model_checkpoint = self._get_chosen_model_checkpoint_path_for_eval(self.__train_output_dir_path) # Ideally all checkpoints should be evaluated
        result = None
        curdir = os.getcwd()
        # we change dir to the eval output directory (for logging) and run the evaluation the checkpoint
        os.chdir(self.__eval_output_dir_path)
        try:
            # copy the setup config to the current dir
            os.system(f"cp {self.__setup_config_path} ./{Path(self.__setup_config_path).name}")
            result = self._run_eval_on_model_checkpoint(
                self.__eval_dataset_config, 
                eval_dataset_base_path=curdir, 
                model_checkpoint=model_checkpoint
            )
        finally:
            os.chdir(curdir)

        # save ret_json as summary file for evaluation
        if result is not None:
            # before saving the summary, we add some attributes by convention
            path = self._get_summary_file_path()
            result.name = Path(path).stem # backwards comp.
            result.model = Path(self.__setup_config_path).stem

            # write model to output file
            with open(path, 'w') as f:
                f.write(result.model_dump_json(indent=4))

        return result

    def load_eval_summary(self) -> Optional[EvalOutputModel]:
        """
            Load the evaluation summary from the output file.
            Returns the model if the file exists, otherwise None.
        """
        with open(self._get_summary_file_path(), 'r') as f:
            return EvalOutputModel.model_validate_json(f.read())