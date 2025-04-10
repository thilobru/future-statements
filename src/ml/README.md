## usage 
Please clone this docker image to get an environment that contains the necessary packages to run the training: `registry.gitlab.com/bdalt/future-statements/training`

The datasets can be found under [datasets/](datasets/) together with additional information. They must be imported via `dvc pull`. 

### Recurrent Neural Network (RNN)

The first Neural Network we implemented is a bidirectional LSTM. It should work as a baseline for our project.

To train the RNN run:

```bash
srun --export=ALL --pty --mem=64g \
--container-name=classifier-training --container-image=bdalt+future-statements+training+latest.sqsh \
--container-mounts=/mnt/ceph:/mnt/ceph --container-writable \
--gres=gpu:1g.5gb \
bash -c "cd $(pwd) && python3 rnn.py train --splitted_dataset_dir ./datasets/splitted_datasets --checkpoint_dir model_checkpoints"
```

To successfully train the model the datasets should be pulled via dvc. Of course it could also be trained on any other `.jsonl` file that has the same structure. There are also various options that can be specified to customize the training behaviour.

### DistilBERT

The second neural network is a Distilbert model. The classification head and optionally the whole model are trained seperately. To train the head run:

```bash
srun --export=ALL --pty --mem=64g \
--container-name=classifier-training --container-image=bdalt+future-statements+training+latest.sqsh \
--container-mounts=/mnt/ceph:/mnt/ceph --container-writable \
--gres=gpu:1g.5gb \
bash -c "cd $(pwd) && python3 distilbert.py train head"
```

There are also various options to customize the training behaviour.

### Parameter Sweeps

We also implemented parameter sweeps for both models with [Weights & Biases](https://wandb.ai/). In case of the DistilBERT model the sweeps are only done on the classification head.

RNN sweep:

```bash
./run_rnn_sweep.sh
```

DistilBERT sweep:

```bash
./run_distilbert_sweep.sh
```
