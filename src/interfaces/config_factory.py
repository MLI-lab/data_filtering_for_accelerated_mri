from src.interfaces.config_models import *

from pydantic_core import from_json
from pydantic_yaml import parse_yaml_file_as

def base_path_config_by_file(file_path: str, is_yml : bool = True) -> BasePathConfigModel:
    if is_yml:
        return parse_yaml_file_as(BasePathConfigModel, file_path)
    else:
        with open(file_path, 'r') as f:
            json_str = f.read()
        return BasePathConfigModel.model_validate(
            from_json(
                json_str,
                allow_partial=False
            )
        )

def train_dataset_config_by_file(file_path: str, is_yml : bool = True) -> TrainDatasetConfigModel:
    if is_yml:
        return parse_yaml_file_as(TrainDatasetConfigModel, file_path)
    else:
        with open(file_path, 'r') as f:
            json_str = f.read()
        return TrainDatasetConfigModel.model_validate(
            from_json(
                json_str,
                allow_partial=False
            )
        )

def eval_dataset_config_by_file(file_path: str, is_yml : bool = True) -> EvalDatasetConfigModel:
    if is_yml:
        return parse_yaml_file_as(EvalDatasetConfigModel, file_path)
    else:
        with open(file_path, 'r') as f:
            json_str = f.read()
        return EvalDatasetConfigModel.model_validate(
            from_json(
                json_str,
                allow_partial=False
            )
        )