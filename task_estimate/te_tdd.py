import numpy as np
import torch
from .te_base import TE_Base, TE_Dynamic_Base
from deir.intrinsic_rewards import build_intrinsic_reward_model


class TE_TemporalDistance(TE_Base):

    task_type = "continuous"

    def __init__(self, env_name, env, tdd_params, batch_size, device) -> None:
        self.env_name = env_name
        n_tasks = 1
        super().__init__(n_tasks)

        self.tdd_params = tdd_params
        self.batch_size = batch_size
        self.device = device
        self.td_model = build_intrinsic_reward_model(
            'tdd',
            observation_space=env.observation_space,
            action_space=env.action_space,
            **self.tdd_params,
            batch_size=self.batch_size,
            device=self.device).to(self.device)

    def predict_task(self, state, info):
        init_obs = info["init_obs"]

        task = self.td_model.get_temporal_distance(obs=init_obs, next_obs=state)
        task_arr = np.array([task])
        return task_arr

    def update(self, replay_buffer, logger, step):
        self.td_model.optimize(replay_buffer=replay_buffer, logger=logger, step=step)


class TE_Dynamic_TemporalDistance(TE_Dynamic_Base):

    task_type = "continuous"

    def __init__(self, env_name, env, tdd_params, batch_size, device) -> None:
        self.env_name = env_name
        n_tasks = 1
        super().__init__(n_tasks)

        self.tdd_params = tdd_params
        self.batch_size = batch_size
        self.device = device
        self.td_model = build_intrinsic_reward_model(
            'tdd',
            observation_space=env.observation_space,
            action_space=env.action_space,
            **self.tdd_params,
            batch_size=self.batch_size,
            device=self.device).to(self.device)

    def get_task_distance(self, state1, state2):
        task_distance = self.td_model.get_temporal_distance(obs=state1, next_obs=state2)
        if not hasattr(task_distance, "shape") or len(task_distance.shape) == 0:
            task_distance = np.array([task_distance])
        else:
            task_distance = task_distance.detach().cpu().numpy()
        return task_distance

    def update(self, replay_buffer, logger, step):
        self.td_model.optimize(replay_buffer=replay_buffer, logger=logger, step=step)

    def save(self, model_dir, step):
        torch.save(self.td_model.state_dict(), f"{model_dir}/td_model_{step}.pt")

    def load(self, model_dir, step):
        self.td_model.load_state_dict(
            torch.load(f"{model_dir}/td_model_{step}.pt", map_location=self.device)
        )
