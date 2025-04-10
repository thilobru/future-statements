from pathlib import Path

# sweeped
default_parameters = {
    'vectorizer_steps': 100_000,
    'vocab_size': 10_000,
    'embedding_dim': 128,
    'lstm_units': 128,
    'learning_rate': 0.0004,
    'dense_layers': [128],
    'dense_dropout': 0.29,
}

default_training_params = {
    'splitted_dataset_dir': Path('datasets/splitted_datasets_00010'),
    'checkpoint_dir': Path('model_checkpoints'),
    'use_wandb': False,
    'num_epochs': 10,
    'examples_per_epoch': 1_000_000,
    'batch_size': 64,
    'shuffle_buffer_size': 10_000,
}

sweep_config = {
    "name": "rnn_future",
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "embedding_dim": {
            "values": [128, 256, 512, 1024]
        },
        "lstm_units": {
            "values": [32, 64, 128]
        },
        "dense_layers": {
            "values": [[128,64],[128],[128,64,32],[64],[]]
        },
        "dense_dropout": {
            "min": 0.0, "max": 0.75,
        },
        "learning_rate": {
            "min": 0.0000001, "max": 0.1, "distribution": "log_uniform_values"
            # according to https://github.com/wandb/client/issues/507
        },
    }
}
