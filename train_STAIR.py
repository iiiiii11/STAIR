#!/usr/bin/env python3
import os
import time
from collections import deque
from pathlib import Path
from pprint import pprint

import hydra
import numpy as np
import torch
import wandb

import utils
from replay_buffer import ReplayBuffer
from replay_buffer_tdd import ReplaybufferForTDD
from reward_model_task_quad import RewardModel_Task_Dynamic
from task_estimate import get_task_distance_predictor
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
                                        name=f"{getDataTimeString()}_{cfg.env}_{cfg.experiment}_{cfg.agent.name}_{cfg.feed_type}_{cfg.max_feedback}_{cfg.segment}_{cfg.seed}")
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
        self.episode_num = 0

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
        if cfg.save_train_video:
            self.env = RecordVideo(
                self.env, f"{Path.cwd()}/train_video",
                episode_trigger=lambda episode_id: episode_id == episode_id,
                video_name_callback=lambda method, episode_id: f"{self.cfg.env}-{self.cfg.agent.name}-{self.episode_num}"
            )
            self.env._max_episode_steps = 1000 if 'dmcontrol' in cfg.env else 5000

        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        cfg.temporal_distance.buffer_params.n_episode_p_update = \
            int(cfg.temporal_distance.buffer_params.n_episode_p_update) * cfg.td_update_freq
        self.td_update_counter = 0
        self.task_model = get_task_distance_predictor(cfg.task_pred,
                                                      env_name=cfg.env, env=self.env,
                                                      tdd_params=cfg.temporal_distance.params,
                                                      batch_size=cfg.agent.params.batch_size,
                                                      device=self.device)

        if cfg.task_pred == "tdd":
            self.replay_buffer = ReplaybufferForTDD(
                self.env.observation_space.shape,
                self.env.action_space.shape,
                int(cfg.replay_buffer_capacity),
                self.device,
                **cfg.temporal_distance.buffer_params,
                discount=self.agent.discount,
                use_offpolicy_data=cfg.temporal_distance.params.offpolicy_data,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.env.observation_space.shape,
                self.env.action_space.shape,
                int(cfg.replay_buffer_capacity),
                self.device)

        self.reward_model = RewardModel_Task_Dynamic(
            ds=self.env.observation_space.shape[0],
            da=self.env.action_space.shape[0],

            device=self.device,
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal,
            data_aug_ratio=cfg.data_aug_ratio,
            data_aug_window=cfg.data_aug_window,
            threshold_u=cfg.threshold_u,
            lambda_u=cfg.lambda_u,
            coef_disagree=cfg.coef_disagree,
            coef_tdminus=cfg.coef_tdminus,
        )

    def evaluate(self):

        average_pred_episode_reward = 0
        average_episode_reward = 0
        success_rate = 0

        for episode in range(self.cfg.num_eval_episodes):
            obs = self.eval_env.reset()
            self.agent.reset()
            done = False
            pred_episode_reward = 0
            episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.eval_env.step(action)

                pred_episode_reward += reward
                episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            average_pred_episode_reward += pred_episode_reward
            average_episode_reward += episode_reward
            if self.log_success:
                success_rate += episode_success

        average_pred_episode_reward /= self.cfg.num_eval_episodes
        average_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0

        self.logger.log('eval/pred_episode_reward', average_pred_episode_reward,
                        self.step)
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                            self.step)
            self.logger.log('train/episode_success', success_rate,
                            self.step)
        self.logger.dump(self.step)

    def learn_reward(self, first_flag=0):

        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:

            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type in (0, 'u'):
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type in (1, 'd'):
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type in (2, 'e'):
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type in (3, 'k'):
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type in (4, 'kd'):
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type in (5, 'ke'):
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            elif self.cfg.feed_type in ('tc', ):
                labeled_queries = self.reward_model.dynamic_task_consistent_sampling(
                    distance_metric_func=self.task_model.get_task_distance)
            elif self.cfg.feed_type in ('tcd', ):
                labeled_queries = self.reward_model.dynamic_task_consistent_disagreement_filter_sampling(
                    distance_metric_func=self.task_model.get_task_distance,
                    disagree_filter_ratio=self.cfg.disagree_filter_ratio)
            elif self.cfg.feed_type in ('tcrm', ):
                labeled_queries = self.reward_model.dynamic_task_consistent_disagreement_real_multiple_sampling(
                    distance_metric_func=self.task_model.get_task_distance)
            else:
                raise NotImplementedError

        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        train_acc = 0
        if self.labeled_feedback > 0:

            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    if self.cfg.data_aug_ratio:
                        train_acc = self.reward_model.semi_train_reward()
                    else:
                        train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)

                if total_acc > 0.97:
                    break

        print(f"Reward function is updated!! ACC: {total_acc} (epoch: {epoch})")
        self.logger.log('train/reward_acc', total_acc, self.step, log_to_mg=False)

    def run(self):

        self.episode_num, pred_episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        episode_reward = 0

        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()
        last_done_step = 0
        task_stat = None

        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            if self.step % int(2e4) == 0:
                self.agent.save(self.work_dir, self.step)
                self.reward_model.save(self.work_dir, self.step)
                self.task_model.save(self.work_dir, self.step)

            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                if self.step > 0 and (last_done_step % self.cfg.eval_frequency) - (self.step % self.cfg.eval_frequency) > 0:
                    self.logger.log('eval/episode', self.episode_num, self.step)
                    self.evaluate()

                self.logger.log('train/pred_episode_reward', pred_episode_reward, self.step)
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)

                if self.log_success:
                    self.logger.log('train/pred_episode_success', episode_success,
                                    self.step)
                    self.logger.log('train/episode_success', episode_success,
                                    self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                pred_episode_reward = 0
                avg_train_true_return.append(episode_reward)
                episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                self.episode_num += 1

                self.logger.log('train/episode', self.episode_num, self.step)
                last_done_step = self.step

                if hasattr(self.replay_buffer, "update_at_episode_end"):
                    self.replay_buffer.update_at_episode_end()

            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):

                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)

                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)

                self.learn_reward(first_flag=1)

                self.replay_buffer.relabel_with_predictor(
                    lambda obses, actions: self.reward_model.r_hat_batch(np.concatenate([obses, actions], axis=-1)))

                self.agent.reset_critic()

                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=self.cfg.reset_update,
                    policy_update=True)

                interact_count = 0

            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:

                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count >= self.cfg.num_interact:

                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)

                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)

                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)

                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(
                            lambda obses, actions: self.reward_model.r_hat_batch(np.concatenate([obses, actions], axis=-1)))
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                self.td_update_counter += 1
                if self.td_update_counter >= self.cfg.td_update_freq:
                    self.task_model.update(self.replay_buffer.get_task_model_buffer(), self.logger, self.step)
                    self.td_update_counter = 0

            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step,
                                            gradient_update=1, K=self.cfg.topK)

            next_obs, reward, done, extra = self.env.step(action)
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            pred_episode_reward += reward_hat
            episode_reward += reward

            if self.log_success:
                episode_success = max(episode_success, extra['success'])

            self.reward_model.add_data(obs=obs, act=action, rew=reward, done=done)
            self.replay_buffer.add(
                obs, action, reward_hat,
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)


@hydra.main(config_path='config/train_STAIR.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
