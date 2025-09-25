import gym
import numpy as np


class InitObsWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.init_obs = None

    def step(self, action):
        observation, reward, done, info = super().step(action)
        info["init_obs"] = self.init_obs
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.init_obs = observation
        self.n_step = 0

        return observation

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__
        else:
            return getattr(self.env, name)
