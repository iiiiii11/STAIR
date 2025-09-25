env_name=$1 
num_train_steps=$2 
group_name=$3
device=$4
seeds=$5
max_feedback=$6 
reward_batch=$7 

common_params="num_train_steps=$num_train_steps agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 num_unsup_steps=9000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3"

common_pebble_params="gradient_update=1 activation=tanh"
query_params="reward_update=10 num_interact=5000 max_feedback=$max_feedback reward_batch=$reward_batch"

time_str=$(date "+%y%m%d-%H%M%S")
wandb_code="wandb=true wandb_group=$time_str-$group_name"
env_code="env=metaworld_$env_name-v2"

metaworld_params=""
pebble_params="$metaworld_params $query_params $common_params $common_pebble_params"

python train_PEBBLE.py $wandb_code device=$device $env_code $pebble_params seed=$seeds feed_type=d