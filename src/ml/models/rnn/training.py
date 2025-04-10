from datetime import datetime

import numpy as np
import tensorflow as tf
import wandb

from .model import _get_vectorization, _get_model
from data_processing import read_data_from_json, dataset_from_file


def run_training(train_file, val_file, test_file, checkpoint_dir, num_epochs, steps_per_epoch, batch_size,
                 shuffle_buffer_size, vectorizer_steps, vocab_size, embedding_dim, lstm_units, learning_rate,
                 dense_layers, dense_dropout, use_wandb
                 ):

    train_ds = dataset_from_file(train_file)

    val_ds = dataset_from_file(val_file)

    test_ds = dataset_from_file(test_file)

    batched_train_ds = (
        train_ds
        .shuffle(shuffle_buffer_size)
        .repeat()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    batched_val_ds = (
        val_ds
        .batch(batch_size)
    )

    batched_test_ds = (
        test_ds
        .batch(batch_size)
    )

    print("adapting test vectorization...")
    text_vectorizer = _get_vectorization(vocab_size)
    text_vectorizer.adapt(
        batched_train_ds.map(lambda text, label: text),
        steps=vectorizer_steps,
    )

    model = _get_model(text_vectorizer, embedding_dim, lstm_units, dense_layers, dense_dropout)

    loss = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
    )

    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            precision,
            recall,
        ],
    )
    callbacks = []

    if checkpoint_dir:
        checkpoint_path = checkpoint_dir / "rnn".format(datetime.now().isoformat())
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='val_loss',  # maybe change to val_precision
        )
        callbacks.append(model_checkpoint_callback)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # maybe change to val_precision
        patience=1,
    )
    callbacks.append(early_stopping_callback)

    if use_wandb:
        wandb_callback = wandb.keras.WandbCallback()
        callbacks.append(wandb_callback)

    model.fit(
        batched_train_ds,
        validation_data=batched_val_ds,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )

    model.evaluate(
        batched_test_ds
    )
