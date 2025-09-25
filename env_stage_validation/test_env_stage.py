import os
import sys

if __name__ == "__main__":
    sys.path.append(".")
    sys.path.append("../")
    if os.path.split(__file__)[0]:
        os.chdir(os.path.join(os.path.split(__file__)[0], "../"))

import hydra
import numpy as np

import utils
from train_SAC import Workspace
from utils import getDataTimeString, save_dict_to_csv

from env_stage_validation.replay_buffer_step import ReplayBufferWithStep
from env_stage_validation.step_classification import StepClassfication

LOAD_MODEL_DIR = ""

ROLLOUT_STEPS = 1e5
DATA_SAVE_DIR = "stage_test"

EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-4
EVAL_DATA_RATE = 0.5

HIDDEN_DIM = 256
HIDDEN_DEPTH = 3

DEVICE = "cuda:0"


def merge_step_data(step_list, reward_list):
    steps = []
    mean_rewards = []
    std_rewards = []
    print(f"max step: {int(step_list.max()) + 1}")
    for i in range(int(step_list.max()) + 1):
        if reward_list[step_list == i].size:
            mean_rewards.append(np.mean(reward_list[step_list == i]))
            std_rewards.append(np.std(reward_list[step_list == i]))
            steps.append(i)
    steps = np.array(steps)
    mean_rewards = np.array(mean_rewards)
    std_rewards = np.array(std_rewards)
    return steps, mean_rewards, std_rewards


def collect_data(env, rollout_steps, replay_buffer: ReplayBufferWithStep,
                 workspace: Workspace):
    step = 0
    obs = env.reset()
    done = False
    episode_reward = 0
    if workspace.log_success:
        episode_success = 0
    episode_step = 0

    while step < rollout_steps:
        with utils.eval_mode(workspace.agent):
            action = workspace.agent.act(obs, sample=False)

        next_obs, reward, done, extra = env.step(action)

        if hasattr(env, "_max_episode_steps") and episode_step + 1 == env._max_episode_steps:
            done_no_max = 0
        else:
            done_no_max = done

        if workspace.log_success:
            episode_success = max(episode_success, extra['success'])

        replay_buffer.add(
            obs, action,
            reward, next_obs, float(done),
            done_no_max, episode_step)

        step += 1

        if done:
            print(f"step: {step}, episode reward: {episode_reward}, episode step: {episode_step}")
            obs = env.reset()
            episode_step = 0
            episode_reward = 0
        else:
            obs = next_obs
            episode_step += 1
            episode_reward += reward


@hydra.main(config_path='../config/train.yaml', strict=True)
def main(cfg):

    workspace = Workspace(cfg)
    if LOAD_MODEL_DIR:
        load_step = cfg.num_train_steps
        workspace.agent.load(model_dir=os.path.join(
            os.getcwd(), "../../../../../" + LOAD_MODEL_DIR), step=int(load_step))
    else:
        workspace.run()

    save_dir = os.path.join(os.getcwd(), "../../../../../", DATA_SAVE_DIR,
                            f"{getDataTimeString()}_{cfg.env}_{cfg.agent.name}_{cfg.seed}")
    os.makedirs(save_dir, exist_ok=True)
    save_path_raw = os.path.join(save_dir, "%s.csv")

    replay_buffer = ReplayBufferWithStep(
        workspace.env.observation_space.shape,
        workspace.action_shape,
        int(cfg.replay_buffer_capacity),
        workspace.device)
    collect_data(workspace.env, ROLLOUT_STEPS, replay_buffer, workspace)

    state_list = replay_buffer.obses
    step_list = replay_buffer.timestep
    reward_list = replay_buffer.rewards

    model_classifier = StepClassfication(cfg, hidden_dim=HIDDEN_DIM,
                                         hidden_depth=HIDDEN_DEPTH, step_range=[step_list.min(), step_list.max()],
                                         eval_data_rate=EVAL_DATA_RATE, lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE)
    model_classifier.train(replay_buffer)

if __name__ == '__main__':
    main()
