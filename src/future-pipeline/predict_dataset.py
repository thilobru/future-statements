import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import AutoTokenizer
import wandb


def load_model():
    api = wandb.Api()
    artifact = api.artifact("future-statements/distilbert-classifier/future-classifier:latest-distilbert", type="model")
    artifact_dir = artifact.download()
    return tf.keras.models.load_model(artifact_dir)


def get_tokens_spec():
    return {
        'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32),
    }


def get_dataset_spec():
    return (
        get_tokens_spec(),  # text for classification
        tf.TensorSpec(shape=(), dtype=tf.string),  # url
        tf.TensorSpec(shape=(), dtype=tf.string),  # text for export
        tf.TensorSpec(shape=(), dtype=tf.string),  # timestamp from website
    )


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer


def predict_passthrough_dataset(dataset_jsonl, batch_size, save_jsonl):
    tokenizer = load_tokenizer()
    model = load_model()

    def _gen():
        with open(dataset_jsonl, mode='r') as ds_file:
            for line in ds_file:
                data = json.loads(line)
                tokenized = tokenizer(data['sentence']).data
                yield tokenized, data['url'], data['sentence'], data['timestamp']

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=get_dataset_spec()
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.padded_batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

    for tokenized, urls, sentences, timestamps in tqdm(ds.as_numpy_iterator()):

        predictions = model.predict_on_batch(tokenized)
        unbatched_predictions = np.reshape(predictions, (-1))
        future_positions = np.argwhere(unbatched_predictions > .5)
        for f_pos in future_positions:
            d = {
                'prediction': predictions[f_pos].item(),
                'sentence': sentences[f_pos].item().decode('utf-8'),
                'url': urls[f_pos].item().decode('utf-8'),
                'timestamp': timestamps[f_pos].item().decode('utf-8'),
            }
            with open(save_jsonl, mode='a') as save_file:
                save_file.write(
                    json.dumps(d, ensure_ascii=False) + '\n'
                )


if __name__ == "__main__":
    dataset_jsonl = Path("filtered_sentences_2022-08-26T14:56:51.966641.jsonl").resolve(strict=True)
    prediction_jsonl = Path("distilbert_predictions_{}.jsonl".format(datetime.now().isoformat())).resolve(strict=False)

    batch_size = 512

    print("dataset: {}".format(Path(dataset_jsonl)))
    print("saved to: {}".format(Path(prediction_jsonl)))
    print("batch size: {}".format(batch_size))
    print("GPUs: {}".format(tf.config.list_physical_devices('GPU')))
    print("GPU name: {}".format(tf.test.gpu_device_name()))
    print("built with cuda: {}".format(tf.test.is_built_with_cuda()))

    predict_passthrough_dataset(
        dataset_jsonl=dataset_jsonl,
        batch_size=batch_size,
        save_jsonl=prediction_jsonl,
    )
