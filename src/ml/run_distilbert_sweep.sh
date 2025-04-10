WORKDIR=$(realpath .) \
SWEEP_SERVER="distil_sweep_server.py" \
SWEEP_AGENT="distil_sweep_agent.py" \
WANDB_ENTITY="future-statements" \
WANDB_PROJECT="distilbert-classifier" \
SWEEPID=$( ./sweep_server.sh ) \
NWORKERS=3 \
./sweep_loop.sh
