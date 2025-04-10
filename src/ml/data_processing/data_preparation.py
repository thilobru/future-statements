from pathlib import Path
import json
import pickle

from sklearn.model_selection import train_test_split
import tensorflow as tf


def read_data_from_json(file_path, max_datapoints=None):
    with open(file_path, mode='r') as f:
        for i, json_obj in enumerate(f):
            if max_datapoints and i >= max_datapoints:
                break
            obj = json.loads(json_obj)
            yield obj['text'], int(obj['is_future'])


def dataset_from_file(sentences_file):
    return tf.data.Dataset.from_generator(
        read_data_from_json,
        args=(str(sentences_file),),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )


def _remove_duplicates(X, y):
    X_new = []
    y_new = []
    d = dict(zip(X, y))
    for x_elem, y_elem in d.items():
        X_new.append(x_elem)
        y_new.append(y_elem)

    return X_new, y_new


def _split_data(X_raw, y, val_size, test_size):
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_raw, y, test_size=val_size, stratify=y)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_train_raw, y_train, test_size=test_size / (1.0 - val_size), stratify=y_train)

    return X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test


def _save_dataset(output_file, texts, labels):
    with open(output_file, mode='w') as f:
        for x, y in zip(texts, labels):
            f.write(
                json.dumps({'text': x, 'is_future': y}, ensure_ascii=False, sort_keys=True) + '\n'
            )


def _save_splitted_data(X_train, y_train, X_val, y_val, X_test, y_test, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    train_file = out_dir / "training.jsonl"
    _save_dataset(train_file, X_train, y_train)
    train_file = out_dir / "validation.jsonl"
    _save_dataset(train_file, X_val, y_val)
    train_file = out_dir / "test.jsonl"
    _save_dataset(train_file, X_test, y_test)


def split_and_save_raw_data(raw_dataset_dir, output_dir, val_size, test_size):
    raw_dir = Path(raw_dataset_dir).resolve(strict=True)
    out_dir = Path(output_dir).resolve(strict=False)
    X_raw = []
    y = []
    print("loading raw data...")
    for f in raw_dir.glob('*.jsonl'):
        for text, label in read_data_from_json(f):
            X_raw.append(text)
            y.append(label)

        print("removing duplicates...")
        X_raw, y = _remove_duplicates(X_raw, y)

        print("splitting data...")
        X_train, y_train, X_val, y_val, X_test, y_test = _split_data(X_raw, y, val_size, test_size)

        print("saving data...")
        _save_splitted_data(X_train, y_train, X_val, y_val, X_test, y_test, out_dir)

    print("done!")
