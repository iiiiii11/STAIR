# STAIR: Addressing Stage Misalignment through Temporal-Aligned Preference Reinforcement Learning

This is the official implementation of STAIR.

## Requirements

Experiments require MuJoCo. Please follow the instructions in the [mujoco-py](https://github.com/openai/mujoco-py) to install.

Install basic packages:

```bash
# install pytorch: https://pytorch.org/get-started/
pip install numpy pandas matplotlib wandb pygame Box2D tensorboard termcolor seaborn ipykernel IPython
pip install gym==0.22.0
pip install hydra-core==0.11.3
pip install "Cython<3"
pip install cloudpickle==2.2.1
pip install imageio==2.19.3
pip install imageio-ffmpeg==0.4.7
pip install Pillow==9.5.0
```

Install customized packages from BPref:

```bash
git clone git@github.com:rll-research/BPref.git
cd BPref
cd custom_dmcontrol
pip install -e .
cd custom_dmc2gym
pip install -e .
```

Install MetaWorld:

```bash
git clone git@github.com:Farama-Foundation/Metaworld.git
cd Metaworld
pip install -e .
```

## Run experiments

All train scripts are stored in `scripts/`. Below are some examples of the scripts. 

Train STAIR for the door-open task.
```bash
bash scripts/metaworld/train_stair_metaworld.sh door-open 1e6 exp-door-open cuda:0 123 5000 50
# bash scripts/metaworld/train_stair_metaworld.sh env_name num_train_steps group_name device seeds max_feedback reward_batch
```

Train PEBBLE for the door-open task.
```bash
bash scripts/metaworld/train_pebble_metaworld.sh door-open 1e6 exp-door-open cuda:0 123 5000 50
# bash scripts/metaworld/train_pebble_metaworld.sh env_name num_train_steps group_name device seeds max_feedback reward_batch
```

Train SAC for the door-open task.
```bash
bash scripts/metaworld/train_sac_metaworld.sh door-open 1e6 exp-door-open cuda:0 123
# bash scripts/metaworld/train_pebble_metaworld.sh env_name num_train_steps group_name device seeds
```

## Baselines

This repo does not include the code for other baseline methods. However, links to the repos of these baseline methods are provided below.
* RIME (also implements MRN and RUNE): https://github.com/CJReinforce/RIME_ICML2024
* QPA: https://github.com/huxiao09/QPA
* MRN: https://github.com/RyanLiu112/MRN
* RUNE: https://github.com/rll-research/rune

## Acknowledgement


This repo is based on [BPref](https://github.com/rll-research/BPref), and benefits from the following repos. Thanks for their wonderful work.
* ETD: https://github.com/Jackory/ETD
* DEIR: https://github.com/swan-utokyo/deir
* dmc2gym: https://github.com/denisyarats/dmc2gym/tree/master
* stable-baseline3: https://github.com/DLR-RM/stable-baselines3
* RoboDesk: https://github.com/google-research/robodesk

