
from collections import defaultdict
import json
import os
import csv
import shutil
import torch
import numpy as np
from termcolor import colored
import wandb

COMMON_TRAIN_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
    ('pred_episode_reward', 'PR', 'float'),
    ('episode_ir', 'IR', 'float'),
    ('total_feedback', 'TF', 'int'),
    ('labeled_feedback', 'LR', 'int'),
    ('noisy_feedback', 'NR', 'int'),
    ('duration', 'D', 'time'),
    ('total_duration', 'TD', 'time'),
    ('td_contrastive_loss', 'TDLOSS', 'float'),
    ('td_logits_pos', 'TDPOS', 'float'),
    ('td_logits_neg', 'TDNEG', 'float'),
    ('td_logits_logsumexp', 'TDLOG', 'float'),
    ('td_categorical_accuracy', 'TDACC', 'float'),
]

COMMON_EVAL_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
    ('pred_episode_reward', 'PR', 'float'),
    ('pred_episode_success', 'PS', 'float'),
]


AGENT_TRAIN_FORMAT = {
    'sac': [
        ('batch_reward', 'BR', 'float'),
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float'),
        ('bc_loss', 'BCLOSS', 'float'),
    ],
}

OTHER_CSV_KEY_TRAIN = [
    "actor_target_entropy", 'critic_norm_entropy_max', 'critic_entropy_min', 'critic_entropy_max',
    'critic_norm_entropy', 'critic_entropy', 'critic_norm_entropy_min',
    'success_rate', 'episode_success', 'pred_episode_success',

    *[f"task{i}" for i in range(10)],
]
OTHER_CSV_KEY_EVAL = [
    'success_rate', 'pred_success_rate',
]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating, other_csv_key=None):
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_name, 'w')
        self._csv_writer = None
        self._other_csv_key = [] if other_csv_key is None else other_csv_key

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        if os.path.exists(file_name):
            os.remove(file_name)
        return file_name

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(
                                                  list(map(lambda x: x[0], self._formating)) + self._other_csv_key),
                                              restval=0.0)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            return f'{key}: {value:04.1f} s'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix, save=True):
        if len(self._meters) == 0:
            return
        if save:
            data = self._prime_meters()
            data['step'] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self,
                 log_dir,
                 save_tb=False,
                 wandb_run=None,
                 wandb_log_video=False,
                 wandb_log_histogram=False,
                 log_frequency=10000,
                 agent='sac'):
        self._log_dir = log_dir
        self._log_frequency = log_frequency
        if save_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                try:
                    shutil.rmtree(tb_dir)
                except:
                    print("logger.py warning: Unable to remove tb directory")
                    pass
            from torch.utils.tensorboard import SummaryWriter
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        self.wandb_run = wandb_run
        self.wandb_log_video = wandb_log_video
        self.wandb_log_histogram = wandb_log_histogram

        assert agent in AGENT_TRAIN_FORMAT
        train_format = COMMON_TRAIN_FORMAT + AGENT_TRAIN_FORMAT[agent]
        self._train_mg = MetersGroup(os.path.join(log_dir, 'train'),
                                     formating=train_format, other_csv_key=OTHER_CSV_KEY_TRAIN)
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'eval'),
                                    formating=COMMON_EVAL_FORMAT, other_csv_key=OTHER_CSV_KEY_EVAL)

    def _should_log(self, step, log_frequency):
        log_frequency = log_frequency or self._log_frequency
        return step % log_frequency == 0

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)
        if self.wandb_run is not None:
            self.wandb_run.log({key: value}, step=step)

    def _try_sw_log_video(self, key, frames, step):
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step, fps=30)
        if self.wandb_run is not None and self.wandb_log_video:
            frames = np.transpose(np.array(frames), (0, 3, 1, 2))
            fps, skip = 6, 8
            self.wandb_run.log({
                'eval/video':
                wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
            })

    def _try_sw_log_histogram(self, key, histogram, step):
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)
        if self.wandb_run is not None and self.wandb_log_histogram:
            self.wandb_run.log({"hist/" + key: histogram}, step=step)

    def log(self, key, value, step, n=1, log_frequency=1, log_to_mg=True):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step)
        if log_to_mg:
            mg = self._train_mg if key.startswith('train') else self._eval_mg
            mg.log(key, value, n)

    def log_param(self, key, param, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias') and hasattr(param.bias, 'data'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_video(self, key, frames, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step, save=True, ty=None):
        if ty is None:
            self._train_mg.dump(step, 'train', save)
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'eval':
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'train':
            self._train_mg.dump(step, 'train', save)
        else:
            raise f'invalid log type: {ty}'
