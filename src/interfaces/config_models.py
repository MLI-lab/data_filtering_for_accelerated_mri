from abc import ABC
from typing import *
from pydantic import BaseModel

## ----------------- Base path configuration ---------------------------- ##
class BasePathConfigModel(BaseModel):
    base_path_train_outputs : str               # base path for training output (e.g. on hdd)
    base_path_eval_outputs : str                # base path for evaluation output
    base_path_eval_summary_file : str           # base path for evaluation summary file
    base_path_train_eval_setup_configs: str     # path to the setup config assets
    base_path_evals_dataset_configs: str        # path to the eval config assets
    base_path_train_dataset_configs: str        # path to the train config assets
    base_path_bart_lib : str                    # path to the bart library

## ----------------- Training dataset configuration ---------------------------- ##
## This defines the training setup, i.e. which datasets we train on in each epoch.
## Similar to data classes this object is generated from the training yaml files.
##  Names, and types are validated when creating this object.
class TrainDatasetConfigModel(BaseModel):
    name : str                    # name of the training setup
    num_epochs : int              # number of epochs to train
    samples_seen : int            # number of samples seen
    epochs_setup : Dict[int, str] # epoch number to dataset name

## ----------------- Evaluation configuration --------------------------------- ##
## This defines the evaluation setup, i.e. which datasets we evaluate on.
class EvalDatasetConfigModel(BaseModel):
    name : str                   # name of the evaluation config
    classic : List[str]          # List of json dataset files
    pathology: List[str]         # List of json dataset files

## ----------------- Evaluation Results --------------------------------------- ##
## This defines what the exact output for each train_eval_setup is,
## in addition to each method performing logging etc. in a experiment directory.
class EvalSingleMetricsModel(BaseModel):
    SSIM : float
    SSIM_normal : float
    PSNR : float
    PSNR_normal : float
    LPIPS : float
    LPIPS_normal : float
    DISTS : float
    DISTS_normal : float

class EvalMetricsModel(BaseModel):
    classic : Dict[str, EvalSingleMetricsModel]
    pathology: Dict[str, EvalSingleMetricsModel]
    
class EvalOutputModel(BaseModel):
    name : str                        # name ?
    uuid : str                        # unique identifier of ?
    model : str                       # currently the name of the model (unet-small), etc.
    creation_date : str               # clear
    eval_metrics : EvalMetricsModel