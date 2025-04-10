from tqdm.auto import tqdm
import tensorflow as tf

from data_processing import read_data_from_json, dataset_from_file


def _tokenize_ds(batched_ds, tokenizer, config_update=None):

    def np_tokenize_batch(str_batch):
        tokenizer_config = {
            "return_tensors": "tf",
            "return_attention_mask": True,
            "return_token_type_ids": True,
            "padding": True,
            "truncation": True,
        }
        if config_update is not None:
            tokenizer_config.update(config_update)  # thanks to python3.8 :(
        decoded = [str(s) for s in str_batch]
        tokenized = tokenizer(decoded, **tokenizer_config)

        return tokenized.input_ids, tokenized.attention_mask, tokenized.token_type_ids

    @tf.function
    def tokenize_batch(str_batch, labels):
        inp_ids, attn_mask, token_t_ids = tf.numpy_function(
            np_tokenize_batch,
            inp=[str_batch],
            Tout=(tf.int32, tf.int32, tf.int32)
        )

        return (
            {"input_ids": tf.convert_to_tensor(inp_ids), "attention_mask": tf.convert_to_tensor(attn_mask)},
            labels
        )

    tokenized_ds = (
        batched_ds
        .map(
            lambda text, label: tokenize_batch(text, label)
        )
    )

    return tokenized_ds


def calc_class_weights(data_generator):
    num_total = 0
    class_occs = {}
    for _, label in data_generator:
        class_occs[label] = class_occs.get(label, 0) + 1
        num_total += 1

    weights = {
        label: 1 / (class_occs[label] / num_total)
        for label in class_occs.keys()
    }

    return weights


def _balance_ds(ds):
    neg_ds = ds.filter(lambda text, label: label == 0)
    pos_ds = ds.filter(lambda text, label: label == 1)

    repeated_pos_ds = pos_ds.repeat()
    repeated_neg_ds = neg_ds.repeat()

    balanced_ds = tf.data.Dataset.sample_from_datasets(
        [repeated_pos_ds, repeated_neg_ds],
        [0.5, 0.5],
    )

    return balanced_ds
