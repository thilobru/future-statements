import tensorflow as tf


def _get_vectorization(vocab_size):

    text_vectorization = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
    )

    return text_vectorization


def _get_model(text_vectorizer, embedding_dim, lstm_units, dense_layers, dense_dropout):

    inp = tf.keras.layers.Input(shape=(1, ), dtype=tf.string)
    vectorized = text_vectorizer(inp)

    vocab_size = len(text_vectorizer.get_vocabulary())

    embedded = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(vectorized)
    encoded = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=lstm_units)
    )(embedded)
    x = tf.keras.layers.Dropout(0.1)(encoded)
    for dl_dim in dense_layers:
        x = tf.keras.layers.Dense(dl_dim, activation='relu')(x)
        x = tf.keras.layers.Dropout(dense_dropout)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inp, out)
