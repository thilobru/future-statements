import argparse
from pathlib import Path

import wandb

from models.distilbert import train_head, head_default_parameters, head_default_training_params, calc_class_weights
from data_processing import read_data_from_json



def sweep_agent():

    params = {
        **head_default_training_params,
        **head_default_parameters,
    }
    wandb.init(config=params)

    hyperparameters = dict(wandb.config)
    hyperparameters.update({"use_wandb": True})

    emb_dir = Path(hyperparameters.pop("splitted_embeddings_dir"))
    emb_train_file = (emb_dir / "training.pkl").resolve(strict=False)
    emb_val_file = (emb_dir / "validation.pkl").resolve(strict=True)
    emb_test_file = (emb_dir / "test.pkl").resolve(strict=True)

    checkpoint_dir = Path(hyperparameters.pop("checkpoint_dir")).resolve(True)

    steps_per_epoch = hyperparameters.pop("examples_per_epoch") // hyperparameters["batch_size"]

    class_weight = {}
    jsonl_for_cw = hyperparameters.pop('jsonl_for_class_weights')
    if hyperparameters['balancing_strategy'] == 'weights':
        class_weight = calc_class_weights(read_data_from_json(jsonl_for_cw))

    train_head(
        train_file=emb_train_file, val_file=emb_val_file, test_file=emb_test_file, steps_per_epoch=steps_per_epoch,
        class_weight=class_weight, checkpoint_dir=checkpoint_dir, **hyperparameters
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a single wandb parameter sweep agent.')
    parser.add_argument('--sweep_id')
    args = parser.parse_args()

    wandb.agent(
        sweep_id=args.sweep_id,
        function=sweep_agent
    )
