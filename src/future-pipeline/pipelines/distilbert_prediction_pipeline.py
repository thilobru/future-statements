from datetime import datetime

import tensorflow as tf
import wandb
from transformers import AutoTokenizer

from future_model_pipeline import FutureModelPipeline


class DistilBertPredictionPipeline(FutureModelPipeline):

    def __init__(self, out_file, max_content_length, min_sentence_length, max_sentence_length, prediction_threshold):
        self.prediction_threshold = prediction_threshold
        super().__init__(out_file, max_content_length, min_sentence_length, max_sentence_length)

    def get_tokens_spec(self):
        return {
            'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        }

    def batch(self, dataset, batchsize):
        return dataset.padded_batch(batchsize, drop_remainder=True)

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        def tokenizer_func(inp):
            return tokenizer(inp).data

        return tokenizer_func

    def get_model(self):
        api = wandb.Api()
        artifact = api.artifact("future-statements/distilbert-classifier/future-classifier:latest-distilbert", type="model")
        artifact_dir = artifact.download()
        return tf.keras.models.load_model(artifact_dir)

    def filter(self, prediction, *args):
        return tf.reshape(prediction > self.prediction_threshold, ())


if __name__ == "__main__":

    out_file = "future_statements_distil_{}.jsonl".format(datetime.now().isoformat())
    p = DistilBertPredictionPipeline(
        out_file=out_file,
        max_content_length=4_000_000,
        min_sentence_length=30,
        max_sentence_length=300,
        prediction_threshold=.5,
    )

    p.run()
