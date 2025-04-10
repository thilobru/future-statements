import pathlib
import json

import click
import wandb

from models.distilbert import load_and_predict, run_training, load_and_evaluate_jsonl, \
    pickle_distilbert_embeddings_from_jsonl, train_head, calc_class_weights, head_default_parameters, \
    head_default_training_params
from data_processing import read_data_from_json


@click.command()
@click.option('--input_file', '-i', type=click.Path(exists=True), required=True)
@click.option('--output_file', '-o', type=click.Path(exists=False), required=True)
@click.option("--model_ref", type=str, default="future-statements/distilbert-classifier/future-classifier:latest-distilbert")
@click.argument('sentences', nargs=-1, type=str)
@click.option("--labelled/--not_labelled", default=False)
def predict(input_file, output_file, model_ref, sentences, labelled):
    sentences = list(sentences)
    true_labels = []
    with open(input_file, mode='r') as f:
        for json_obj in f:
            obj = json.loads(json_obj)
            sentences.append(obj['text'])
            if labelled:
                true_labels.append(int(obj['is_future']))

    raw, rounded = load_and_predict(sentences, model_ref)

    with open(output_file, mode='w') as f:
        for s, l_true, p_raw, p_rounded in zip(sentences, true_labels, raw, rounded):
            print("{}: true: {}, pred: {} (raw: {})".format(s, l_true, p_rounded, p_raw))
            f.write(
                json.dumps({'text': s, 'true_label': l_true, 'pred_label': int(p_rounded), 'pred_label_raw': p_raw.item()},
                ensure_ascii=False, sort_keys=True) + '\n'
            )


@click.command()
@click.option('--jsonl_file', '-f', type=click.Path(exists=True))
@click.option('--model_ref', type=str, default="future-statements/distilbert-classifier/future-classifier:latest-distilbert")
def evaluate(jsonl_file, model_ref):
    results = load_and_evaluate_jsonl(jsonl_file, model_ref)
    print(results)


@click.command('head')
@click.option("--splitted_embeddings_dir", '-e', type=click.Path(exists=True, path_type=pathlib.Path, resolve_path=True), default=head_default_training_params['splitted_embeddings_dir'])
@click.option("--jsonl_for_class_weights", '-w', type=click.Path(exists=True, path_type=pathlib.Path, resolve_path=True), default=head_default_training_params['jsonl_for_class_weights'])
@click.option("--checkpoint_dir", '-c', type=click.Path(path_type=pathlib.Path, resolve_path=True), default=head_default_training_params['checkpoint_dir'])
@click.option("--use_wandb/--no_wandb", default=head_default_training_params['use_wandb'])
@click.option("--wandb_project", type=str, default=head_default_training_params['wandb_project'])
@click.option("--wandb_entity", type=str, default=head_default_training_params['wandb_entity'])
@click.option("--num_epochs", type=int, default=head_default_training_params['num_epochs'])
@click.option("--examples_per_epoch", type=int, default=head_default_training_params['examples_per_epoch'])
@click.option("--batch_size", type=int, default=head_default_training_params['batch_size'])
@click.option("--shuffle_buffer_size", type=int, default=head_default_training_params['shuffle_buffer_size'])
@click.option("--input_dropout_rate", type=int, default=head_default_parameters['input_dropout_rate'])
@click.option("--dense_layers", "-d", multiple=True, default=head_default_parameters['dense_layers'])
@click.option("--dense_dropout_rate", type=float, default=head_default_parameters['dense_dropout_rate'])
@click.option("--learning_rate", type=float, default=head_default_parameters['learning_rate'])
@click.option("--balancing_strategy", type=click.Choice(['sampling', 'weights', 'no_balancing'], case_sensitive=False), default=head_default_training_params['balancing_strategy'])
def t_head(**kwargs):
    emb_dir = kwargs.pop("splitted_embeddings_dir")
    emb_train_file = (emb_dir / "training.pkl").resolve(strict=False)
    emb_val_file = (emb_dir / "validation.pkl").resolve(strict=True)
    emb_test_file = (emb_dir / "test.pkl").resolve(strict=True)

    steps_per_epoch = kwargs.pop("examples_per_epoch") // kwargs["batch_size"]
    wandb_project = kwargs.pop('wandb_project')
    wandb_entity = kwargs.pop('wandb_entity')
    if kwargs['use_wandb']:
        wandb.init(project=wandb_project, entity=wandb_entity, config=kwargs)

    class_weight = {}
    jsonl_for_cw = kwargs.pop('jsonl_for_class_weights')
    if kwargs['balancing_strategy'] == 'weights':
        class_weight = calc_class_weights(read_data_from_json(jsonl_for_cw))


    train_head(
        train_file=emb_train_file, val_file=emb_val_file, test_file=emb_test_file, steps_per_epoch=steps_per_epoch,
        class_weight=class_weight, **kwargs
    )


@click.command('all')
@click.option("--splitted_dataset_dir", '-j', type=click.Path(exists=True, path_type=pathlib.Path, resolve_path=True), required=True)
@click.option("--checkpoint_dir", '-c', type=click.Path(path_type=pathlib.Path, resolve_path=True), required=True)
@click.option("--head_ref", type=str, default="future-statements/distilbert-classifier/future-head:latest")
@click.option("--use_wandb/--no_wandb", default=False)
@click.option("--wandb_project", type=str, default="distilbert-classifier")
@click.option("--wandb_entity", type=str, default="future-statements")
@click.option("--num_epochs", type=int, default=5)
@click.option("--examples_per_epoch", type=int, default=1_000_000)
@click.option("--batch_size", type=int, default=16)
@click.option("--shuffle_buffer_size", type=int, default=10_000)
@click.option("--learning_rate", type=float, default=1e-5)
@click.option("--balancing_strategy", type=click.Choice(['sampling', 'weights', 'no_balancing'], case_sensitive=False), default='no_balancing')
def t_all(**kwargs):
    ds_dir = kwargs.pop('splitted_dataset_dir')
    train_file = (ds_dir / "training.jsonl").resolve(strict=True)
    val_file = (ds_dir / "validation.jsonl").resolve(strict=True)
    test_file = (ds_dir / "test.jsonl").resolve(strict=True)

    steps_per_epoch = kwargs.pop("examples_per_epoch") // kwargs["batch_size"]

    wandb_project = kwargs.pop('wandb_project')
    wandb_entity = kwargs.pop('wandb_entity')
    if kwargs['use_wandb']:
        wandb.init(project=wandb_project, entity=wandb_entity, config=kwargs)

    run_training(
        train_file=train_file, val_file=val_file, test_file=test_file, steps_per_epoch=steps_per_epoch,
        **kwargs
    )


@click.group()
def train(**kwargs):
    pass


train.add_command(t_head)
train.add_command(t_all)

@click.command()
@click.option("--splitted_dataset_dir", "-i", type=click.Path(exists=True, path_type=pathlib.Path, resolve_path=True))
@click.option("--target_dir", "-o", type=click.Path(exists=False, path_type=pathlib.Path, resolve_path=True))
@click.option("--batch_size", "-b", type=int, default=256)
def embed(splitted_dataset_dir, target_dir, batch_size):

    target_dir.mkdir(exist_ok=True)
    print("pickling training...")
    train_in_file = (splitted_dataset_dir / 'training.jsonl').resolve(strict=True)
    train_out_file = (target_dir / 'training.pkl').resolve(strict=False)
    pickle_distilbert_embeddings_from_jsonl(batch_size, train_in_file, train_out_file)

    print("pickling validation...")
    val_in_file = (splitted_dataset_dir / 'validation.jsonl').resolve(strict=True)
    val_out_file = (target_dir / 'validation.pkl').resolve(strict=False)
    pickle_distilbert_embeddings_from_jsonl(batch_size, val_in_file, val_out_file)

    print("pickling test...")
    test_in_file = (splitted_dataset_dir / 'test.jsonl').resolve(strict=True)
    test_out_file = (target_dir / 'test.pkl').resolve(strict=False)
    pickle_distilbert_embeddings_from_jsonl(batch_size, test_in_file, test_out_file)


@click.group()
def cli():
    pass


cli.add_command(predict)
cli.add_command(evaluate)
cli.add_command(train)
cli.add_command(embed)


if __name__ == "__main__":
    cli()
