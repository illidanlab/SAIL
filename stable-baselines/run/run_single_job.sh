#!/bin/sh
prefix="/mnt/hdd2/judy/reward_shaping/baselines"
if [ "$#" -le 3 ]
then
    echo "Usage:     ./run_lfrs.sh <env> <algo> <task> <seed> (n-timesteps, optional) <total-jobs, optional> <n-episodes, default1>"
    echo "Example:   ./run_lfrs.sh HalfCheetah-v2 td3 gail4-lfd 1 6 4"
    exit
else
    env=$1
    algo=$2
    task=$3
    seed=$4
    if [ "$#" -ge 5 ]
	then
	n_job=$5
    fi
    if [ "$#" -ge 6 ]
	then
	n_episodes=$6
    fi
    n_time="-1"
    name=""
    
    logdir="$prefix/$task/$algo/$env/rank$seed"
    mkdir -p logdir

    if [ "$algo" = "td3" ]
	then
	program="train_pofd.py"
    elif [ "$algo" = "dice" ]
	then
	program="train_dice.py"
    else
	program="train_sail.py"
    fi
    echo "python $program --env $env --seed $seed --algo $algo --log-dir $logdir --task $task --n-timesteps $n_time --n-jobs $n_job --n-episodes $n_episodes"
    #python $program --env $env --seed $seed --algo $algo --log-dir $logdir --task $task --n-timesteps $n_time --n-jobs $n_job --n-episodes $n_episodes
fi

