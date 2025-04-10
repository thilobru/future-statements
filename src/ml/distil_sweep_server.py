import wandb
from models.distilbert import head_sweep_config

if __name__ == "__main__":
    sweep_id = wandb.sweep(head_sweep_config)

    print(sweep_id)
