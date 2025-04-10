import numpy as np

import wandb
import tensorflow as tf

from .model import _get_tokenizer
from .data_preprocessing import _tokenize_ds
from data_processing import dataset_from_file


def _load_model(model_ref="future-statements/distilbert-classifier/future-classifier:latest-distilbert"):
    api = wandb.Api()
    artifact = api.artifact(model_ref, type="model")
    artifact_dir = artifact.download()
    return tf.keras.models.load_model(artifact_dir)


def load_and_predict(sentences, model_ref):
    model = _load_model(model_ref)
    tokenized_sentences = _tokenize_data(sentences).data
    preds = _predict(model, tokenized_sentences)
    reshaped = preds.reshape(-1)
    rounded = np.around(reshaped)

    return reshaped, rounded


def _load_classification_head(head_ref="future-statements/distilbert-classifier/future-head:latest"):
    api = wandb.Api()
    artifact = api.artifact(head_ref, type="model")
    artifact_dir = artifact.download()
    return tf.keras.models.load_model(artifact_dir)


def load_and_evaluate_jsonl(eval_file, model_ref):
    model = _load_model(model_ref)
    tokenizer = _get_tokenizer()
    eval_ds = _tokenize_ds(dataset_from_file(eval_file).batch(8), tokenizer=tokenizer)
    return model.evaluate(eval_ds)


def _predict(model, to_predict):
    raw_predictions = model.predict(
        to_predict,
        batch_size=16,
    )
    return raw_predictions


def _tokenize_data(sentences):
    tokenizer_config = {
        "return_tensors": "tf",
        "return_attention_mask": True,
        "return_token_type_ids": True,
        "padding": True,
        "truncation": True,
    }
    tokenizer = _get_tokenizer()
    return tokenizer(sentences, **tokenizer_config)
