env_name=$1 
num_train_steps=$2 
group_name=$3
device=$4
seeds=$5
max_feedback=$6 
reward_batch=$7 

quad_params="num_interact=30000 agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001"
walker_params="num_interact=20000 agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005"

case "${env_name}" in
    "walker_run"|"cheetah_run")
        env_params=$walker_params
        ;;
    *)
        env_params=$quad_params
        ;;
esac

common_params="num_train_steps=$num_train_steps num_unsup_steps=9000"
common_pebble_params="gradient_update=1 activation=tanh"
query_params="reward_update=50 max_feedback=$max_feedback reward_batch=$reward_batch"

time_str=$(date "+%y%m%d-%H%M%S")
wandb_code="wandb=true wandb_group=$time_str-$group_name"
env_code="env=$env_name"

dmc_params="$env_params"
pebble_params="$dmc_params $query_params $common_params $common_pebble_params"

python train_PEBBLE.py $wandb_code device=$device $env_code $pebble_params seed=$seeds feed_type=d