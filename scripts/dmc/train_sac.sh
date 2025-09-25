env_name=$1 
num_train_steps=$2
group_name=$3
device=$4
seeds=$5

common_params="num_train_steps=$num_train_steps num_unsup_steps=9000"

time_str=$(date "+%y%m%d-%H%M%S")
wandb_code="wandb=true wandb_group=$time_str-$group_name"
env_code="env=$env_name"
sac_params="$common_params"

python train_SAC.py $wandb_code device=$device $env_code $sac_params seed=$seeds