import numpy as np
import torch
import torch.nn.functional as F
import utils
from agent.conv_ac import DiscreteConvQ
from torch import nn


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class DoubleQDiscreteCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim, hidden_dim, action_dim, hidden_depth)
        self.Q2 = utils.mlp(obs_dim, hidden_dim, action_dim, hidden_depth)
        self.action_dim = action_dim

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        assert obs.size(0) == action.size(0)

        q1 = self.Q1(obs)
        q2 = self.Q2(obs)

        if action.dim() != q1.dim():
            action = action.unsqueeze(1)

        q1_action = torch.gather(q1, -1, action.to(torch.int64))
        q2_action = torch.gather(q2, -1, action.to(torch.int64))

        self.outputs['q1'] = q1_action
        self.outputs['q2'] = q2_action

        return q1_action, q2_action

    def get_value(self, obs: torch.Tensor):
        q1 = self.Q1(obs)
        q2 = self.Q2(obs)
        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class DiscreteConvCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()

        self.Q1 = DiscreteConvQ(obs_shape, action_dim)
        self.Q2 = DiscreteConvQ(obs_shape, action_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        assert obs.size(0) == action.size(0)

        q1 = self.Q1(obs)
        q2 = self.Q2(obs)

        if action.dim() != q1.dim():
            action = action.unsqueeze(1)

        q1_action = torch.gather(q1, -1, action.to(torch.int64))
        q2_action = torch.gather(q2, -1, action.to(torch.int64))

        return q1_action, q2_action

    def get_value(self, obs: torch.Tensor):
        q1 = self.Q1(obs)
        q2 = self.Q2(obs)
        return q1, q2

    def log(self, logger, step):
        pass
