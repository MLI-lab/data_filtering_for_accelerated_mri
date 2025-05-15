from ....interfaces.dataset_models import DatasetModel
from .SliceDataset import SliceDataset
from .VolumeDataset import VolumeDataset

from pydantic_core import from_json
from pydantic_yaml import parse_yaml_file_as

def dataset_model_by_file(file_path: str, is_yml : bool = True, allow_partial: bool = True) -> DatasetModel:
    if is_yml:
        return parse_yaml_file_as(DatasetModel, file_path)
    else:
        with open(file_path, 'r') as f:
            json_str = f.read()
        return DatasetModel.model_validate(
            from_json(
                json_str,
                allow_partial=allow_partial,
            )
        )

def get_dataset(path_to_json, transform, augment_data=False):
    """
        Creates a torch dataset object following the fastMRI format.

        First, we validate the json dataset config file, and then read the dataset as 2D slice dataset or 3D volume dataset.
    """
    dataset_model = dataset_model_by_file(path_to_json, is_yml=False, allow_partial=True)
    if dataset_model.dataset_is_3d:
        assert not augment_data, "Augmentation is not implemented for 3D datasets currently."
        return VolumeDataset(dataset_model, transform=transform, readout_dim_fftshift_cor=dataset_model.readout_dim_fftshift_cor)
    else:
        return SliceDataset(dataset_model, transform=transform, augment_data=augment_data)