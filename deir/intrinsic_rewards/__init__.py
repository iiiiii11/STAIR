'''
from deir: https://github.com/swan-utokyo/deir
'''

from .tdd import TDDModel

IR_MODELS = {
    'tdd': TDDModel,
}


def build_intrinsic_reward_model(ir_name, *args, **kwargs):
    if ir_name in IR_MODELS:
        return IR_MODELS[ir_name](*args, **kwargs)
    else:
        return None
