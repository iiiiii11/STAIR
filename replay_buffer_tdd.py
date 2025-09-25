from typing import Generator, Optional

import numpy as np
import torch

from deir.buffers.type_aliases import ReplayBufferSamples
from replay_buffer import ReplayBuffer


class ReplaybufferForTDD(ReplayBuffer):
    def __init__(self, obs_shape, action_shape, capacity, device, discount, use_offpolicy_data, n_batch_p_update, n_episode_p_update, window=1):
        super().__init__(obs_shape, action_shape, capacity, device, window)
        self.discount = discount
        self.future_obses = np.zeros((self.capacity,) + self.obs_shape, dtype=self.obs_dtype)

        self.aux_obses = np.zeros((self.capacity,) + self.obs_shape, dtype=self.obs_dtype)
        self.aux_future_obses = np.zeros((self.capacity,) + self.obs_shape, dtype=self.obs_dtype)
        self.aux_pos = 0
        self.aux_full = False

        self.idx_last_future = 0
        self.idx_last_last_future = 0

        self.use_offpolicy_data = use_offpolicy_data
        self.n_batch_p_update = n_batch_p_update
        self.n_episode_p_update = n_episode_p_update
        self.episode_cnt = 0
        self.tdd_buffer_wrapper = ReplaybufferForTDD_Wrapper(self)

    def get_sub_buffer(self, key, start, end):
        buffer = self.__dict__[key]
        if start <= end:
            return buffer[start:end]
        else:
            return np.concatenate([buffer[start, :], buffer[:end]])

    def compute_future_states(self):
        if self.idx > self.idx_last_future:
            T = self.idx - self.idx_last_future
        else:
            T = self.capacity - self.idx_last_future + self.idx
        assert T > 0
        ranges = np.full((T, ), T, dtype=np.int32)
        last_one = T - 1

        not_dones = self.get_sub_buffer("not_dones", self.idx_last_future, self.idx)
        obses = self.get_sub_buffer("obses", self.idx_last_future, self.idx)
        for t in reversed(range(T)):

            if not not_dones[t]:
                last_one = t
            ranges[t] = last_one - t

        seeds = np.random.rand(*ranges.shape)
        if self.discount == 0:
            intervals = np.ones_like(seeds, dtype=ranges.dtype)
        elif self.discount == 1:
            intervals = np.ceil(seeds * ranges).astype(int)
        else:
            intervals = np.log(1 - (1 - self.discount**ranges) * seeds) / np.log(self.discount)
        intervals = np.minimum(np.ceil(intervals).astype(int), ranges)

        future_obses = obses[intervals + np.arange(T)]
        if self.idx > self.idx_last_future:
            self.future_obses[self.idx_last_future:self.idx] = future_obses
        else:
            self.future_obses[self.idx_last_future:] = future_obses[:self.capacity - self.idx_last_future]
            self.future_obses[:self.idx] = future_obses[self.capacity - self.idx_last_future:]
        self.idx_last_last_future = self.idx_last_future
        self.idx_last_future = self.idx

    def add_to_aux_buffer(self):

        dims = len(self.obses.shape)

        obses_new = self.obses[:self.idx]
        future_obses_new = self.future_obses[:self.idx]

        same_elements = np.all(future_obses_new == obses_new, axis=tuple(range(1, dims)))
        length = obses_new[~same_elements].shape[0]

        if self.aux_pos + length > self.capacity:
            trunc = self.capacity - self.aux_pos
            self.aux_obses[self.aux_pos:] = obses_new[~same_elements][:trunc].copy()
            self.aux_future_obses[self.aux_pos:] = future_obses_new[~same_elements][:trunc].copy()
            self.aux_obses[:length - trunc] = obses_new[~same_elements][trunc:].copy()
            self.aux_future_obses[:length - trunc] = future_obses_new[~same_elements][trunc:].copy()
            self.aux_full = True
            self.aux_pos = length - trunc
        else:
            self.aux_obses[self.aux_pos:self.aux_pos + length] = obses_new[~same_elements].copy()
            self.aux_future_obses[self.aux_pos:self.aux_pos + length] = future_obses_new[~same_elements].copy()
            self.aux_pos = self.aux_pos + length

    def update_at_episode_end(self):
        self.episode_cnt += 1
        if self.episode_cnt < self.n_episode_p_update:
            return
        self.episode_cnt = 0
        self.compute_future_states()
        if self.use_offpolicy_data:
            self.add_to_aux_buffer()

    def get_task_model_buffer(self):
        return self.tdd_buffer_wrapper


class ReplaybufferForTDD_Wrapper:
    def __init__(self, buffer: ReplaybufferForTDD):
        self.buffer = buffer
        self.device = self.buffer.device
        self.use_offpolicy_data = self.buffer.use_offpolicy_data
        self.n_batch_p_update = self.buffer.n_batch_p_update

    def get(self, batch_size: Optional[int] = None):
        if not self.use_offpolicy_data:

            if self.buffer.idx_last_future >= self.buffer.idx_last_last_future:
                indices = np.random.permutation(self.buffer.idx_last_future - self.buffer.idx_last_last_future) + \
                    self.buffer.idx_last_last_future
            else:
                indices_raw = np.concatenate([np.arange(self.buffer.idx_last_last_future, self.buffer.capacity),
                                              np.arange(0, self.buffer.idx_last_future)], axis=0)
                indices = np.random.permutation(indices_raw)
            start_idx = 0
            while start_idx < indices.shape[0]:

                yield self._get_onpolicy_samples(indices[start_idx: start_idx + batch_size]), None
                start_idx += batch_size
        else:
            for i in range(self.n_batch_p_update):
                yield None, self._get_offpolicy_samples(batch_size)

    def _get_offpolicy_samples(self, batch_size) -> ReplayBufferSamples:
        upper_bound = self.buffer.capacity if self.buffer.aux_full else self.buffer.aux_pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        data = (
            self.buffer.aux_obses[batch_inds],
            self.buffer.aux_future_obses[batch_inds]
        )
        samples = tuple(map(lambda x: self.to_torch(x, copy=False), data))
        return ReplayBufferSamples(*samples)

    def _get_onpolicy_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        data = (
            self.buffer.obses[batch_inds],
            self.buffer.future_obses[batch_inds]
        )
        samples = tuple(map(lambda x: self.to_torch(x, copy=False), data))
        return ReplayBufferSamples(*samples)

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """ (from stable_baseline3)
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)
