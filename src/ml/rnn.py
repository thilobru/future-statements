import pathlib

import click
import wandb

from models.rnn import load_and_predict, load_and_evaluate_jsonl, run_training, default_parameters, sweep_config, \
    default_training_params
from data_processing import read_data_from_json


@click.command()
@click.option('--jsonl-file', '-f', type=click.Path(exists=True))
@click.argument('sentences', nargs=-1, type=str)
def predict(jsonl_file, sentences):
    sentences = list(sentences)
    if jsonl_file:
        for text, _ in read_data_from_json(jsonl_file):
            sentences.append(text)
    raw, rounded = load_and_predict(sentences)

    for s, p_raw, p_rounded in zip(sentences, raw, rounded):
        print("{}: {} (raw: {})".format(s, p_rounded, p_raw))


@click.command()
@click.argument('jsonl_file', nargs=1, type=click.Path(exists=True))
def evaluate(jsonl_file):
    load_and_evaluate_jsonl(jsonl_file)


@click.command()
@click.option("--splitted_dataset_dir", "-i", type=click.Path(exists=True, path_type=pathlib.Path, resolve_path=True), default=default_training_params['splitted_dataset_dir'])
@click.option("--checkpoint_dir", "-m", type=click.Path(path_type=pathlib.Path, resolve_path=True), default=default_training_params['checkpoint_dir'])
@click.option("--use_wandb/--no_wandb", default=default_training_params['use_wandb'])
@click.option("--num_epochs", type=int, default=default_training_params['num_epochs'])
@click.option("--examples_per_epoch", type=int, default=default_training_params['examples_per_epoch'])
@click.option("--batch_size", type=int, default=default_training_params['batch_size'])
@click.option("--shuffle_buffer_size", type=int, default=default_training_params['shuffle_buffer_size'])
@click.option("--vectorizer_steps", type=int, default=default_parameters['vectorizer_steps'])
@click.option("--vocab_size", type=int, default=default_parameters['vocab_size'])
@click.option("--embedding_dim", type=int, default=default_parameters['embedding_dim'])
@click.option("--lstm_units", type=int, default=default_parameters['lstm_units'])
@click.option("--learning_rate", type=float, default=default_parameters['learning_rate'])
@click.option("--dense_layers", "-d", multiple=True, default=default_parameters['dense_layers'])
@click.option("--dense_dropout", type=float, default=default_parameters['dense_dropout'])
def train(**kwargs):

    ds_dir = kwargs.pop('splitted_dataset_dir')
    train_file = (ds_dir / "training.jsonl").resolve(strict=True)
    val_file = (ds_dir / "validation.jsonl").resolve(strict=True)
    test_file = (ds_dir / "test.jsonl").resolve(strict=True)

    steps_per_epoch = kwargs.pop('examples_per_epoch') // kwargs['batch_size']

    if kwargs['use_wandb']:
        wandb.init(project="rnn-classifier", entity="future-statements", config=kwargs)

    run_training(train_file, val_file, test_file, steps_per_epoch=steps_per_epoch, **kwargs)


@click.command('server')
def sweep_server():
    sweep_id = wandb.sweep(sweep_config)
    print(sweep_id)
    return sweep_id


@click.command('agent')
def sweep_agent():
    params = {
        **default_training_params,
        **default_parameters,
        'use_wandb': True,
    }
    wandb.init(config=params)
    hyperparams = dict(wandb.config)

    ds_dir = pathlib.Path(hyperparams.pop('splitted_dataset_dir'))
    train_file = (ds_dir / "training.jsonl").resolve(strict=True)
    val_file = (ds_dir / "validation.jsonl").resolve(strict=True)
    test_file = (ds_dir / "test.jsonl").resolve(strict=True)

    steps_per_epoch = hyperparams.pop('examples_per_epoch') // hyperparams['batch_size']

    run_training(train_file, val_file, test_file, steps_per_epoch=steps_per_epoch, **hyperparams)


@click.group()
def sweep():
    pass


sweep.add_command(sweep_agent)
sweep.add_command(sweep_server)


@click.group()
def cli():
    pass


cli.add_command(predict)
cli.add_command(evaluate)
cli.add_command(train)
cli.add_command(sweep)


if __name__ == "__main__":
    cli()
