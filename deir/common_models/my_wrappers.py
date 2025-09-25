import math
import operator
from functools import reduce

import gym
import numpy as np
from gym import spaces
from gym.core import ObservationWrapper, Wrapper

class FlatImageWrapper(ObservationWrapper):
    """
    Flat observed images into one flat array and discard the mission str
    """

    def __init__(self, env):
        super().__init__(env)

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize,),
            dtype="uint8",
        )

    def observation(self, obs):
        image = obs["image"]
        obs = image.flatten()

        return obs


class ImageWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        imgSpace = env.observation_space.spaces["image"]
        self.observation_space = imgSpace

    def observation(self, obs):
        image = obs["image"]

        return image


class StateNoChangePenalty(Wrapper):
    """
    penalize the agent if the action make no differnce to the obs
    """

    def __init__(self, env):
        super().__init__(env)
        self.last_img = None
        self.penalty = -0.01

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        img = obs["image"]
        if np.linalg.norm(self.last_img - img) < 1e-5:
            reward += self.penalty
        self.last_img = img

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_img = obs["image"]
        return obs


class TransposeImageWrapper(Wrapper):
    """
    Re-order channels, from HxWxC to CxHxW.
    It is required for PyTorch convolution layers.
    """

    def __init__(self, env):
        observation_space = self.transpose_space(env.observation_space)
        super(TransposeImageWrapper, self).__init__(env, observation_space=observation_space)

    @staticmethod
    def transpose_space(observation_space: spaces.Box) -> spaces.Box:
        """
        Transpose an observation space (re-order channels).
        """
        width, height, channels = observation_space.shape
        new_shape = (channels, width, height)
        return spaces.Box(low=0, high=255, shape=new_shape, dtype=observation_space.dtype)

    @staticmethod
    def transpose_image(image: np.ndarray) -> np.ndarray:
        """
        Transpose an image or batch of images (re-order channels).
        """
        if len(image.shape) == 3:
            return np.transpose(image, (2, 0, 1))
        return np.transpose(image, (0, 3, 1, 2))

    def step(self):
        observations, rewards, dones, infos = self.env.step()
        return self.transpose_image(observations), rewards, dones, infos

    def reset(self) -> np.ndarray:
        """
        Reset all environments
        """
        return self.transpose_image(self.env.reset())

