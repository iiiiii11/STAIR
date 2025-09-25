import datetime
import math
import os
import random
from functools import wraps
from typing import Dict, Optional, Union
import csv

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch import distributions as pyd
from torch import nn
from torch.nn import RNNCellBase


def getDataTimeString():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')[2:]


def option_params(condition, **params):
    if condition:
        return params
    return {}


def NotVerified(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        raise Exception(f"function {func.__name__} is not verified")

    return wrapped_func


def hydra_config_to_dict(hydra_config: DictConfig, _prefix=""):
    ret_dict = {}
    for k in hydra_config.keys():
        try:
            v = hydra_config.get(k, default_value='???')
        except Exception as e:
            v = str(e)
        if isinstance(v, DictConfig):
            nv = hydra_config_to_dict(v, f"{_prefix}{k}.")
            ret_dict.update(nv)
        else:
            ret_dict[f"{_prefix}{k}"] = v
    return ret_dict


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def np_onehot(c, n_class):
    ret = np.zeros(n_class)
    ret[int(c)] = 1
    return ret


def np_onehot_vec(v: np.ndarray, n_class):
    ret = np.zeros((v.shape[0], n_class))
    ret[np.arange(v.shape[0]), v] = 1
    return ret


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):

        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):

        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self):
        return torch.sqrt(self.var)


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


class RunningMeanStd(object):
    """
    Implemented based on:
    - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    - https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/mpi_util.py#L179-L214
    - https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
    """

    def __init__(self, epsilon=1e-4, momentum=None, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.eps = epsilon
        self.momentum = momentum

    def clear(self):
        self.__init__(self.eps, self.momentum)

    @staticmethod
    def update_ema(old_data, new_data, momentum):
        if old_data is None:
            return new_data
        return old_data * momentum + new_data * (1.0 - momentum)

    def update(self, x):
        batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        if self.momentum is None or self.momentum < 0:
            self.update_from_moments(batch_mean, batch_var, batch_count)
        else:
            self.mean = self.update_ema(self.mean, batch_mean, self.momentum)
            new_var = np.mean(np.square(x - self.mean))
            self.var = self.update_ema(self.var, new_var, self.momentum)
            self.std = np.sqrt(self.var)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.std = np.sqrt(new_var)
        self.count = new_count


class CustomGRUCell(RNNCellBase):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,

                 bias: bool = True,
                 device=None,
                 dtype=None,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomGRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3, **factory_kwargs)
        self.norm_i = nn.BatchNorm1d(hidden_size * 3, momentum=0.1)
        self.norm_h = nn.BatchNorm1d(hidden_size * 3, momentum=0.1)
        self.norm_n = nn.BatchNorm1d(hidden_size, momentum=0.1)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        return self.gru_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )

    def gru_cell(self, inputs, hidden, w_ih, w_hh, b_ih, b_hh):
        gi = self.norm_i(torch.mm(inputs, w_ih.t()))
        gh = self.norm_h(torch.mm(hidden, w_hh.t()))
        if self.bias:
            gi = gi + b_ih
            gh = gh + b_hh
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(self.norm_n(i_n + resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy


def obs_as_tensor(obs: Union[np.ndarray, Dict[str, np.ndarray]], device: torch.device):
    """
    Moves the observation to the given device.
    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray):
        return torch.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: torch.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")


def jaccard_index(interval1, interval2, use_neg_intersect=False):
    '''return intersect / union; if use_neg_intersect, return - gap / (union + gap) if not interact'''
    intersection_start = np.maximum(interval1[0], interval2[0])
    intersection_end = np.minimum(interval1[1], interval2[1])

    if not use_neg_intersect:
        intersection_length = np.maximum(0, intersection_end - intersection_start)
    else:
        intersection_length = intersection_end - intersection_start
    union_length = (interval1[1] - interval1[0]) + (interval2[1] - interval2[0]) - intersection_length
    jaccard_index = intersection_length / (union_length + 1e-5)

    return jaccard_index


def save_dict_to_csv(data, filename):
    headers = list(data.keys())
    rows = zip(*data.values())

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def load_csv_to_dict(filename):
    with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        data = {header: [] for header in headers}
        for row in reader:
            for i, value in enumerate(row):
                data[headers[i]].append(float(value))
    return data
