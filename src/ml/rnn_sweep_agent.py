import argparse
from pathlib import Path

import wandb

from models.rnn import run_training, default_parameters, default_training_params


def sweep_agent():
    params = {
        **default_training_params,
        **default_parameters,
    }
    wandb.init(config=params)
    hyperparams = dict(wandb.config)
    hyperparams.update({'use_wandb': True})

    ds_dir = Path(hyperparams.pop('splitted_dataset_dir')).resolve(strict=True)
    print(ds_dir)
    train_file = (ds_dir / "training.jsonl").resolve(strict=True)
    val_file = (ds_dir / "validation.jsonl").resolve(strict=True)
    test_file = (ds_dir / "test.jsonl").resolve(strict=True)

    checkpoint_dir = Path(hyperparams.pop("checkpoint_dir")).resolve(True)

    steps_per_epoch = hyperparams.pop('examples_per_epoch') // hyperparams['batch_size']

    run_training(train_file, val_file, test_file, checkpoint_dir=checkpoint_dir, steps_per_epoch=steps_per_epoch, **hyperparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a single wandb parameter sweep agent.')
    parser.add_argument('--sweep_id')
    args = parser.parse_args()

    wandb.agent(sweep_id=args.sweep_id,
                function=sweep_agent)
