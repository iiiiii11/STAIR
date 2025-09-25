import numpy as np
from .te_base import TE_Base


class TE_Time(TE_Base):
    
    task_type = "continuous"

    def __init__(self, env_name, env) -> None:
        self.env_name = env_name
        n_tasks = 1  
        super().__init__(n_tasks)

    def predict_task(self, state, info):
        n_step = info["n_step"]
        task_arr = np.array([n_step])
        return task_arr
