env_name=$1 
num_train_steps=$2
group_name=$3
device=$4
seeds=$5

common_params="num_train_steps=$num_train_steps agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 num_unsup_steps=9000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3"

time_str=$(date "+%y%m%d-%H%M%S")
wandb_code="wandb=true wandb_group=$time_str-$group_name"
env_code="env=$env_name"
sac_params="$common_params"

python train_SAC.py $wandb_code device=$device $env_code $sac_params seed=$seeds