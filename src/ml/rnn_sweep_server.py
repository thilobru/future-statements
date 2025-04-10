import wandb
from models.rnn import sweep_config

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config)

    print(sweep_id)
