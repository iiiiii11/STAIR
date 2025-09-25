#!/usr/bin/env python3
import os
import pickle as pkl
import time
from pathlib import Path
from pprint import pprint

import hydra
import torch
import wandb

import utils
from replay_buffer import ReplayBuffer
from utils import getDataTimeString, hydra_config_to_dict
from utils.logger import Logger
from utils.make_env import make_env
from utils.my_record_video import RecordVideo


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        if cfg.wandb:
            self.wandb_run = wandb.init(project=cfg.wandb_name, config=hydra_config_to_dict(cfg),
                                        name=f"{getDataTimeString()}_{cfg.env}_{cfg.experiment}_{cfg.agent.name}_{cfg.seed}")
        else:
            self.wandb_run = None

        self.cfg = cfg
        pprint(dict(cfg))
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            wandb_run=self.wandb_run,
            wandb_log_video=cfg.wandb_log_video,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)

        self.device = torch.device(cfg.device)
        self.log_success = False
        self.step = 0

        self.env = make_env(cfg, is_eval=False)
        self.eval_env = make_env(cfg, is_eval=True)

        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        cfg.agent.params.obs_dim = self.obs_dim
        cfg.agent.params.action_dim = self.action_dim

        if 'metaworld' in cfg.env or 'robodesk' in cfg.env:
            self.log_success = True

        self.env.seed(cfg.seed)
        self.eval_env.seed(cfg.seed)

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.action_shape = self.env.action_space.shape if self.env.action_space.shape else (1,)

        if cfg.save_video:
            self.eval_env = RecordVideo(
                self.eval_env, f"{Path.cwd()}/video",

                episode_trigger=lambda episode_id: episode_id % self.cfg.num_eval_episodes == 1,
                video_name_callback=lambda method, episode_id: f"{self.cfg.env}-{self.cfg.agent.name}-{self.step}"
            )

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.action_shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        meta_file = os.path.join(self.work_dir, 'metadata.pkl')
        pkl.dump({'cfg': self.cfg}, open(meta_file, "wb"))

    def evaluate(self):
        average_episode_reward = 0
        if self.log_success:
            success_rate = 0

        for episode in range(self.cfg.num_eval_episodes):
            obs = self.eval_env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0

            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.eval_env.step(action)
                episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            average_episode_reward += episode_reward
            if self.log_success:
                success_rate += episode_success

        average_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0

        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)

        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                            self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        start_time = time.time()
        fixed_start_time = time.time()
        last_done_step = 0

        while self.step < self.cfg.num_train_steps:
            if self.step % int(2e4) == 0:
                self.agent.save(self.work_dir, self.step)

            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    self.logger.log('train/total_duration',
                                    time.time() - fixed_start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                if self.step > 0 and (last_done_step % self.cfg.eval_frequency) - (self.step % self.cfg.eval_frequency) > 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                                    self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                last_done_step = self.step

            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps) and self.cfg.num_unsup_steps > 0:

                self.agent.reset_critic()

                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=self.cfg.reset_update,
                    policy_update=True)
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step,
                                            gradient_update=1, K=self.cfg.topK)

            next_obs, reward, done, extra = self.env.step(action)

            done = float(done)
            if hasattr(self.env, "_max_episode_steps") and episode_step + 1 == self.env._max_episode_steps:
                done_no_max = 0
            else:
                done_no_max = done

            episode_reward += reward

            if self.log_success:
                episode_success = max(episode_success, extra['success'])

            self.replay_buffer.add(
                obs, action,
                reward, next_obs, done,
                done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

        self.agent.save(self.work_dir, self.step)


@hydra.main(config_path='config/train_SAC.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
