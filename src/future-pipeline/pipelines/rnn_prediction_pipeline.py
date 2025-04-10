from datetime import datetime

import tensorflow as tf
import wandb

from future_model_pipeline import FutureModelPipeline


class RnnPredictionPipeline(FutureModelPipeline):

    def __init__(self, out_file, max_content_length, min_sentence_length, max_sentence_length, prediction_threshold):
        self.prediction_threshold = prediction_threshold
        super().__init__(out_file, max_content_length, min_sentence_length, max_sentence_length)

    def get_tokens_spec(self):
        return tf.TensorSpec(shape=(), dtype=tf.string)

    def get_tokenizer(self):

        def tokenizer(text):
            return text

        return tokenizer

    def get_model(self):
        api = wandb.Api()
        artifact = api.artifact("future-statements/rnn-classifier/future-classifier:latest-rnn", type="model")
        artifact_dir = artifact.download()
        return tf.keras.models.load_model(artifact_dir)

    def filter(self, prediction, *args):
        return tf.reshape(prediction > self.prediction_threshold, ())


if __name__ == "__main__":

    out_file = "future_statements_{}.jsonl".format(datetime.now().isoformat())
    p = RnnPredictionPipeline(
        out_file=out_file,
        max_content_length=4_000_000,
        min_sentence_length=30,
        max_sentence_length=500,
        prediction_threshold=.5,
    )

    p.run()
