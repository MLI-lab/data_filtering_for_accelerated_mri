# data_filt_registry.py

__DATA_FILTERING_SETUPS__ = {}

def register_data_filtering_setup():
    def wrapper(cls):
        name = cls.__name__
        if __DATA_FILTERING_SETUPS__.get(name, None):
            raise NameError(f"DataFilteringSetup {name} is already registered!")
        __DATA_FILTERING_SETUPS__[name] = cls
        return cls
    return wrapper

def get_data_filtering_setup(name: str, **kwargs):
    if __DATA_FILTERING_SETUPS__.get(name, None) is None:
        raise NameError(f"DataFilteringSetup {name} is not defined!")
    return __DATA_FILTERING_SETUPS__[name](**kwargs)

def list_data_filtering_setups():
    return list(__DATA_FILTERING_SETUPS__.keys())

def list_data_filtering_setups_str():
    return "\n".join(list_data_filtering_setups())

from src.data_filt_setups.HeuristicFiltering.energy_filtering import EnergyFiltering
from src.data_filt_setups.HeuristicFiltering.EdgeSparsityDetection import EdgeSparsityDetection