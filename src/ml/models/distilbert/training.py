import tensorflow as tf
# import tensorflow_addons as tfa
import wandb

from .data_preprocessing import _tokenize_ds, _balance_ds, calc_class_weights
from .embeddings import dataset_from_pickled_embeddings, load_pickled_embeddings
from .model import _build_model, _build_classification_head, _get_tokenizer, _add_transformer, TRANSFORMER_LAYER
from .prediction import _load_classification_head
from data_processing import dataset_from_file, read_data_from_json


def _compile_model(model, learning_rate):

    loss = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            # tfa.metrics.F1Score(num_classes=1, threshold=0.5),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
        ],
    )


def train_head(
        train_file, val_file, test_file, checkpoint_dir, use_wandb, num_epochs, steps_per_epoch,
        batch_size, shuffle_buffer_size, input_dropout_rate, dense_layers, dense_dropout_rate, learning_rate,
        balancing_strategy, class_weight
):

    train_ds = dataset_from_pickled_embeddings(train_file)
    val_ds = dataset_from_pickled_embeddings(val_file)
    test_ds = dataset_from_pickled_embeddings(test_file)

    if balancing_strategy == 'weights':
        print("Balancing strategy: weights")
        print(class_weight)
    elif balancing_strategy == 'sampling':
        print("Balancing strategy: sampling")
        train_ds = _balance_ds(train_ds)
    elif balancing_strategy == 'no_balancing':
        print("Balancing strategy: no balancing")
    else:
        raise Exception

    print("class weight:", class_weight)

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
        .prefetch(tf.data.AUTOTUNE)
    )

    batched_test_ds = (
        test_ds
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = _build_classification_head(input_dropout_rate, dense_layers, dense_dropout_rate)

    _compile_model(model, learning_rate)

    callbacks = []
    checkpoint_path = checkpoint_dir / "distilbert_head"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor='val_loss',  # alternative: val_loss
    )
    callbacks.append(model_checkpoint_callback)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # alternative: val_loss
        patience=1,
    )
    callbacks.append(early_stopping_callback)

    if use_wandb:
        wandb_callback = wandb.keras.WandbCallback()
        callbacks.append(wandb_callback)

    history_frozen = model.fit(
        batched_train_ds,
        validation_data=batched_val_ds,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    model.evaluate(
        batched_test_ds
    )

    return model


def run_training(
        train_file, val_file, test_file, checkpoint_dir, use_wandb, num_epochs, steps_per_epoch, batch_size,
        shuffle_buffer_size, learning_rate, balancing_strategy, head_ref,
):
    print("loading head: {}".format(head_ref))
    model_head = _load_classification_head(head_ref=head_ref)

    tokenizer = _get_tokenizer()

    train_ds = dataset_from_file(train_file)
    val_ds = dataset_from_file(val_file)
    test_ds = dataset_from_file(test_file)

    class_weight = {}
    if balancing_strategy == 'weights':
        print("Balancing strategy: weights")
        class_weight = calc_class_weights(
            read_data_from_json(train_file)
        )
        print("class weight:", class_weight)
    elif balancing_strategy == 'sampling':
        print("Balancing strategy: sampling")
        train_ds = _balance_ds(train_ds)
    elif balancing_strategy == 'no_balancing':
        print("Balancing strategy: no balancing")
    else:
        raise Exception

    batched_train_ds = (
        train_ds
        .shuffle(shuffle_buffer_size)
        .repeat()
        .batch(batch_size)
    )

    batched_val_ds = (
        val_ds
        .batch(batch_size)
    )

    batched_test_ds = (
        test_ds
        .batch(batch_size)
    )

    tokenized_train_ds = _tokenize_ds(batched_train_ds, tokenizer).prefetch(tf.data.AUTOTUNE)
    tokenized_val_ds = _tokenize_ds(batched_val_ds, tokenizer).prefetch(tf.data.AUTOTUNE)
    tokenized_test_ds = _tokenize_ds(batched_test_ds, tokenizer).prefetch(tf.data.AUTOTUNE)

    model = _add_transformer(model_head)

    model.get_layer(TRANSFORMER_LAYER).trainable = True

    callbacks = []

    checkpoint_path = checkpoint_dir / "distilbert"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor='val_loss',  # alternative: val_loss
    )
    callbacks.append(model_checkpoint_callback)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # alternative: val_loss
        patience=1,
    )
    callbacks.append(early_stopping_callback)

    if use_wandb:
        wandb_callback = wandb.keras.WandbCallback()
        callbacks.append(wandb_callback)

    _compile_model(model, learning_rate)

    history_unfrozen = model.fit(
        tokenized_train_ds,
        validation_data=tokenized_val_ds,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    model.evaluate(
        tokenized_test_ds
    )

