from typing import Tuple
from src.interfaces.config_models import *

import os
import sys

from pathlib import Path
import yaml

"""
    We have two output folders:
    -----------------------------------------
    1. for training (logs, configs used, model checkpoints etc):
        - identified by the setup name, train dataset name and the setup config file name (but not the eval dataset)
        - contains large model files and is not tracked by git
        - *not* stored in the working directory
    2. for evaluation (logs, configs used, example reconstructions etc.):
        - identified by the setup name, train dataset name, eval dataset config name, and the setup config file name
        - contains logs and potentially some larger files, such as example reconstructions (take more space in 3D)
        - *not* stored in the working directory

    Moreover, there is a tracked json-result file
    -----------------------------------------
    3. evaluation (currently a single json file):
        - is identified by the setup name, train dataset name, eval dataset name and the setup config file name (as in the eval output folder)
        - stored in the working directory
        - tracked by git, to have backups and versioning
"""

def setup_train_output_dir_path(
    base_path_config : BasePathConfigModel,
    train_eval_setup_name : str,
    setup_config_path : str,
    train_dataset_config : TrainDatasetConfigModel,
    create_folders : bool = True
) -> str:
    # defining the name of the folders and files
    train_output_name = train_eval_setup_name + "_" + train_dataset_config.name + "_" + Path(setup_config_path).stem

    # create full paths by concateining the previous names with the base paths
    train_output_path = os.path.join(base_path_config.base_path_train_outputs, train_output_name)

    if create_folders:
        if not os.path.isdir(train_output_path):
            os.makedirs(train_output_path)

    return train_output_path

def setup_eval_output_dir_path(
    base_path_config : BasePathConfigModel,
    train_eval_setup_name : str,
    setup_config_path : str,
    train_dataset_config : TrainDatasetConfigModel,
    eval_dataset_config : EvalDatasetConfigModel,
    create_folders : bool = True
) -> str:
    # defining the name of the folders and files
    eval_output_name = train_eval_setup_name + '_' + train_dataset_config.name + "_" + eval_dataset_config.name + "_" + Path(setup_config_path).stem 

    # create full paths by concateining the previous names with the base paths
    eval_output_dir_path = os.path.join(base_path_config.base_path_eval_outputs, eval_output_name)

    if create_folders:
        if not os.path.isdir(eval_output_dir_path):
            os.makedirs(eval_output_dir_path)

    return eval_output_dir_path

def setup_eval_summary_file_path(
    base_path_config : BasePathConfigModel,
    train_eval_setup_name : str,
    setup_config_path : str,
    train_dataset_config : TrainDatasetConfigModel,
    eval_dataset_config : EvalDatasetConfigModel,
    extra_info : str = "",
    file_ext : str = ".json",
    create_folders : bool = True
) -> str:
    # defining the name of the folders and files
    eval_output_name = train_eval_setup_name + '_' + train_dataset_config.name + "_" + eval_dataset_config.name + "_" + Path(setup_config_path).stem + "_" + extra_info

    # create full paths by concateining the previous names with the base paths
    eval_summary_path = os.path.join(base_path_config.base_path_eval_summary_file, eval_output_name + file_ext)

    if create_folders:
        if not os.path.isdir(Path(eval_summary_path).parents[0]):
            os.makedirs(Path(eval_summary_path).parents[0])

    return eval_summary_path

def import_bart_from_path_model(
    base_path_config : BasePathConfigModel
):
    sys.path.insert(0, os.path.join(base_path_config.base_path_bart_lib, "python"))
    os.environ['TOOLBOX_PATH'] = base_path_config.base_path_bart_lib
    import bart # testing
