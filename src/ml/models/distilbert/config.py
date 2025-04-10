from pathlib import Path

# ethereal-sweep-11
head_default_parameters = {
    "input_dropout_rate": 0.15,
    "dense_layers": [128, 64, 16],
    "dense_dropout_rate": 0.15,
    "learning_rate": 1e-4,
}

head_default_training_params = {
    "splitted_embeddings_dir": Path('datasets/splitted_embeddings_00010'),
    "jsonl_for_class_weights": None,
    "checkpoint_dir": Path('model_checkpoints'),
    "use_wandb": False,
    'wandb_project': 'distilbert-classifier',
    'wandb_entity': 'future-statements',
    "num_epochs": 10,
    "examples_per_epoch": 1_000_000,
    "batch_size": 16,
    "shuffle_buffer_size": 10_000,
    "balancing_strategy": 'no_balancing'
}

head_sweep_config = {
    "name": "distilbert_future",
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "dense_layers": {
            "values": [[128, 64], [128], [128, 64, 32], [128, 64, 16], [512], [512, 256], [64], []]
        },
        "input_dropout_rate": {
            "min": 0.0, "max": 0.5,
        },
        "dense_dropout_rate": {
            "min": 0.0, "max": 0.75,
        },
        "learning_rate": {
            "min": 0.0000001, "max": 0.1, "distribution": "log_uniform_values"
        },
    }
}
