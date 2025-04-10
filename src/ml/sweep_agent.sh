#!/bin/bash
#
#SBATCH --job-name=future_sweep
#
#SBATCH --mem-per-gpu=5g
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.5gb
#SBATCH --gpu-bind=single:1
#SBATCH --nodes=1
#SBATCH --output=/dev/null
#SBATCH -e agent-%j.err
#SBATCH -o agent-%j.out

srun \
--export=ALL \
--container-image=./bdalt+future-statements+training+latest.sqsh \
--container-mounts=/mnt/ceph:/mnt/ceph \
--container-name=sweep_container-$1 \
bash -c " cd $WORKDIR && PYTHONPATH=. WANDB_ENTITY=$WANDB_ENTITY WANDB_PROJECT=$WANDB_PROJECT WANDB_CONSOLE=auto python3 $SWEEP_AGENT --sweep_id=$SWEEPID "