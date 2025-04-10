import pickle

import tensorflow as tf
from tqdm.auto import tqdm

from .model import _get_transformer, _get_tokenizer
from .data_preprocessing import _tokenize_ds
from data_processing import dataset_from_file


def _get_embeddings_from_str_ds(tokenizer, model, str_ds, batch_size):

    tokenized_ds = _tokenize_ds(
        str_ds.batch(batch_size),
        tokenizer,
        # {"padding": 'max_length'}
    ).prefetch(tf.data.AUTOTUNE)

    for feat_batch, lbl_batch in tokenized_ds:
        batch_emb = model(feat_batch).last_hidden_state  # (bs, seq_len, dim)
        for item_emb, item_lbl in zip(batch_emb[:, 0, :], lbl_batch[:]):  # (1, dim) -> only CLS token
            yield item_emb, item_lbl


def pickle_distilbert_embeddings_from_jsonl(batch_size, jsonl_file, out_file):

    tokenizer = _get_tokenizer()
    model = _get_transformer()
    str_ds = dataset_from_file(jsonl_file)

    embed_generator = tqdm(
        _get_embeddings_from_str_ds(
            tokenizer, model, str_ds, batch_size
        )
    )

    embed_generator.set_description(str(out_file))

    with open(out_file, mode='wb') as f:
        for item_emb, item_lbl in embed_generator:
            pickle.dump((item_emb, item_lbl), f)


def load_pickled_embeddings(pickled_embeddings_file):

    with open(pickled_embeddings_file, mode='rb') as f:
        while True:
            try:
                emb, lbl = pickle.load(f)
                yield emb, lbl.numpy()
            except EOFError as e:
                break


def dataset_from_pickled_embeddings(pickled_embeddings_file):
    return tf.data.Dataset.from_generator(
        load_pickled_embeddings,
        args=(str(pickled_embeddings_file),),
        output_signature=(
            tf.TensorSpec(shape=(768, ), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )
