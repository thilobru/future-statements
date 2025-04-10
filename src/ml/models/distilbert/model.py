import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer


MODEL = "distilbert-base-uncased"
TRANSFORMER_LAYER = "distilbert"  # if not calling .distilbert on DistilbertModel:  tf_distil_bert_model


def _build_model(input_dropout_rate, dense_layers, dense_dropout_rate):

    transformer = _get_transformer()
    input_ids = tf.keras.layers.Input(shape=(None, ), name="input_ids", dtype="int32")
    attention_mask = tf.keras.layers.Input(shape=(None, ), name="attention_mask", dtype="int32")

    # call to (distil)bert because of: https://github.com/keras-team/keras/issues/14345
    distilbert_output = transformer.distilbert(input_ids=input_ids, attention_mask=attention_mask)
    # according to https://github.com/huggingface/transformers/blob/v4.21.1/src/transformers/models/distilbert/modeling_tf_distilbert.py#L699
    hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
    pooled_output = hidden_state[:, 0]  # (bs, dim)

    x = tf.keras.layers.Dropout(input_dropout_rate)(pooled_output)

    for out_dim in dense_layers:
        x = tf.keras.layers.Dense(out_dim, activation="relu")(x)
        x = tf.keras.layers.Dropout(dense_dropout_rate)(x)

    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model([input_ids, attention_mask], out)

    return model


def _build_classification_head(input_dropout_rate, dense_layers, dense_dropout_rate):
    cls_input = tf.keras.layers.Input(shape=(768, ), name='cls_input', dtype='float32')

    x = tf.keras.layers.Dropout(input_dropout_rate)(cls_input)

    for out_dim in dense_layers:
        x = tf.keras.layers.Dense(out_dim, activation="relu")(x)
        x = tf.keras.layers.Dropout(dense_dropout_rate)(x)

    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model([cls_input], out)

    return model


def _add_transformer(classification_head):
    input_ids = tf.keras.layers.Input(shape=(None,), name="input_ids", dtype="int32")
    attention_mask = tf.keras.layers.Input(shape=(None,), name="attention_mask", dtype="int32")

    transformer = _get_transformer()
    t_out = transformer.distilbert(input_ids=input_ids, attention_mask=attention_mask)
    t_cls = t_out.last_hidden_state[:, 0, :]

    h_out = classification_head(t_cls)

    model = tf.keras.Model([input_ids, attention_mask], h_out)

    return model


def _get_transformer():
    return TFAutoModel.from_pretrained(MODEL)


def _get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)
