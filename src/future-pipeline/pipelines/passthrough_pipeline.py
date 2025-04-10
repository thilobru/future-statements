from datetime import datetime
import tensorflow as tf

from future_model_pipeline import FutureModelPipeline


class PassthroughPipeline(FutureModelPipeline):

    def __init__(self, out_file, max_content_length, min_sentence_length, max_sentence_length):
        super().__init__(out_file, max_content_length, min_sentence_length, max_sentence_length)

    def get_model(self):
        return None

    def predict(self, model_input, *args):
        return tf.ones((self.BATCHSIZE,)), *args

    @tf.function
    def filter(self, *args):
        return True


if __name__ == "__main__":

    out_file = "filtered_sentences_{}.jsonl".format(datetime.now().isoformat())
    p = PassthroughPipeline(
        out_file=out_file,
        max_content_length=4_000_000,
        min_sentence_length=30,
        max_sentence_length=300,
    )

    p.run()