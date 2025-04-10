import pathlib
import json
import click
import random

from data_processing import split_and_save_raw_data


@click.command()
@click.option("--raw_dataset_dir", "-i", type=click.Path(exists=True, path_type=pathlib.Path, resolve_path=True))
@click.option("--output_dir", "-o", type=click.Path(exists=False, path_type=pathlib.Path, resolve_path=True))
@click.option("--val_size", "-v", type=float, default=0.1)
@click.option("--test_size", "-t", type=float, default=0.1)
def split(raw_dataset_dir, output_dir, val_size, test_size):

    split_and_save_raw_data(
        raw_dataset_dir=raw_dataset_dir, output_dir=output_dir, val_size=val_size, test_size=test_size
    )

@click.command('sample')
@click.option("--input_file", "-i", type=click.Path(exists=True, path_type=pathlib.Path, resolve_path=True))
@click.option("--output_file", "-o", type=click.Path(exists=False, path_type=pathlib.Path, resolve_path=True))
@click.option("--n_lines", "-n", type=int, default=1000)
def extract_random(input_file, output_file, n_lines):
    sentences = []
    with open(input_file, mode='r') as f:
        for json_obj in f:
            obj = json.loads(json_obj)
            sentences.append(obj['text'])
    
    sentences = list(set(sentences))
    random_sentences = random.sample(sentences, k=n_lines)

    with open(output_file, mode='w') as f:
        for s in random_sentences:
            f.write(
                json.dumps({'text': s}, ensure_ascii=False, sort_keys=True) + '\n'
            )


@click.group()
def cli():
    pass


cli.add_command(split)
cli.add_command(extract_random)


if __name__ == "__main__":
    cli()
