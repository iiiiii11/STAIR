from .te_base import TE_Base, TE_Dynamic_Base
from .te_time_oracle import TE_Time
from .te_tdd import TE_TemporalDistance, TE_Dynamic_TemporalDistance

TASK_PREDICTORS = {
    "tdd": TE_TemporalDistance,
    "time_step": TE_Time,
}


def get_task_predictor(predictor_name, *args, **kwargs) -> TE_Base:
    return TASK_PREDICTORS[predictor_name](*args, **kwargs)


TASK_DISTANCE_PREDICTORS = {
    "tdd": TE_Dynamic_TemporalDistance,
}


def get_task_distance_predictor(predictor_name, *args, **kwargs) -> TE_Dynamic_Base:
    return TASK_DISTANCE_PREDICTORS[predictor_name](*args, **kwargs)
