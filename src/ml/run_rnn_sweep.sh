WORKDIR=$(realpath .) \
SWEEP_SERVER="rnn_sweep_server.py" \
SWEEP_AGENT="rnn_sweep_agent.py" \
WANDB_ENTITY="future-statements" \
WANDB_PROJECT="rnn-classifier" \
SWEEPID=$( ./sweep_server.sh ) \
NWORKERS=1 \
./sweep_loop.sh
