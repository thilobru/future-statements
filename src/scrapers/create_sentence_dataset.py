import json
import pathlib

import nltk
nltk.download('punkt')


def create_dataset_from_jsonl(jsonl_path, text_key):

    with open(jsonl_path, mode='r') as data_file:

        sentences = []

        for data_line in data_file:

            data_dict = json.loads(data_line)
            sentences += [
                {'text': s, 'label': None} 
                for s in nltk.tokenize.sent_tokenize(data_dict[text_key])
            ]

        return sentences
    

def save_data(data, out_file):

    with open(out_file, mode='x') as out_f:

        for datapoint in data:

            out_f.write(json.dumps(datapoint) + '\n')


if __name__ == "__main__":

    data_dir = pathlib.Path(__file__).parent / 'data'
    inet_data = data_dir / 'internet_future_data.jsonl'
    futurist_data = data_dir / 'futurist_speaker.jsonl'
    
    train_inetdb_file = data_dir / 'training_inetdb.jsonl'
    train_fs_file = data_dir / 'training_fs.jsonl'

    data_inet = create_dataset_from_jsonl(inet_data, 'excerpt')
    data_fs = create_dataset_from_jsonl(futurist_data, 'content')

    save_data(data_inet, train_inetdb_file)
    save_data(data_fs, train_fs_file)
