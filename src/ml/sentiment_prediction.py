import json
from pathlib import Path

from transformers import pipeline
from tqdm.auto import tqdm

def load_data(input_file):
    with open(input_file, mode='r') as f:
        for line in f:
            d = json.loads(line)
            yield d


def predict_file(to_predict):
    pipe = pipeline(model='j-hartmann/sentiment-roberta-large-english-3-classes', device=0, batch_size=128)
    lines = list(load_data(to_predict))
    for prediction, line in zip(pipe(list(map(lambda x: x['sentence'], lines))), lines):
        yield {
            'sent_label': prediction['label'],
            'sent_score': prediction['score'],
            **line
        }


def predict_sentiments(in_file, out_file):
    with open(out_file, mode='x') as out:
        for d in tqdm(predict_file(in_file)):
            out.write(
                json.dumps(
                    d,
                    ensure_ascii=False,
                ) + '\n'
            )

if __name__ == "__main__":
    input_file = Path(__file__).parent.parent / 'final_datasets/future_statements_rnn.jsonl'
    output_file = Path(__file__).parent.parent / 'final_datasets/future_statements_rnn_sentiments.jsonl'

    predict_sentiments(input_file, output_file)