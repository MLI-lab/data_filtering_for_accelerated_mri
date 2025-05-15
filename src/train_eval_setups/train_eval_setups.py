
__BASE_TRAIN_EVAL_SETUPS__ = {}

#def register_train_eval_setup(name: str):
    #def wrapper(cls):
        #if __BASE_TRAIN_EVAL_SETUPS__.get(name, None):
            #raise NameError(f"BaseEvalExp {name} is already registered!")
        #__BASE_TRAIN_EVAL_SETUPS__[name] = cls
        #return cls
    #return wrapper

def register_train_eval_setup():
    def wrapper(cls):
        name = cls.__name__
        if __BASE_TRAIN_EVAL_SETUPS__.get(name, None):
            raise NameError(f"BaseTrainEvalSetup {name} is already registered!")
        __BASE_TRAIN_EVAL_SETUPS__[name] = cls
        return cls
    return wrapper

def get_train_eval_setup(name: str, **kwargs):
    if __BASE_TRAIN_EVAL_SETUPS__.get(name, None) is None:
        raise NameError(f"BaseTrainEvalSetup {name} is not defined!")
    return __BASE_TRAIN_EVAL_SETUPS__[name](**kwargs)

def list_train_eval_setups():
    return list(__BASE_TRAIN_EVAL_SETUPS__.keys())

def list_train_eval_setups_str():
    return "\n".join(list_train_eval_setups())

from src.train_eval_setups.end_to_end.end2end_setup import End2EndSetup
from src.train_eval_setups.diff_model_recon.diff_model_recon_setup import DiffModelVariationSetup
