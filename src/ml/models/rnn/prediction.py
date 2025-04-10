import numpy as np
import tensorflow as tf
import wandb

from data_processing import dataset_from_file


def _load_model():
    api = wandb.Api()

    artifact = api.artifact("future-statements/rnn-classifier/future-classifier:latest-rnn", type="model")
    artifact_dir = artifact.download()
    return tf.keras.models.load_model(artifact_dir)


def _predict(model, sentences):
    sentences = np.array(sentences)
    preds = model.predict(sentences)
    return preds


def load_and_predict(sentences):
    model = _load_model()
    preds = _predict(model, sentences)
    reshaped = preds.reshape(-1)
    rounded = np.around(reshaped)

    return reshaped, rounded


def _predict_jsonl(model, predict_file):
    predict_ds = dataset_from_file(predict_file).batch(8)
    preds = model.predict(predict_ds)
    reshaped = preds.reshape(-1)
    rounded = np.around(reshaped)

    return reshaped, rounded


def load_and_predict_jsonl(predict_file):
    model = _load_model()
    reshaped, rounded = _predict_jsonl(model, predict_file)

    return reshaped, rounded


def load_and_evaluate_jsonl(eval_file):
    model = _load_model()
    eval_ds = dataset_from_file(eval_file).batch(8)
    model.evaluate(eval_ds)
