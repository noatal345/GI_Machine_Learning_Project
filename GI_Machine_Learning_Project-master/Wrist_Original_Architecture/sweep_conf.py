# Import the W&B Python Library and log into W&B
import wandb
from main_for_wrist import main_wrist

wandb.login()


def main():
    wandb.init()
    test_acc = main_wrist(wandb.config)
    wandb.log({"Test accuracy": test_acc})


# 2: Define the search space
sweep_configuration = {
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'Test accuracy'},
    'parameters':
        {
            'lr': {'values': [0.00001, 0.0001, 0.001]},
            'epoch': {'values': [10, 20, 50, 100]},
            'shape': {'values': ["64_128", "256_512", "1024_2048"]},
            'num_of_measurements': {'values': [512, 1024, 2048, 4096, 8192]},
            'batch_size': {'values': [16, 32, 64, 128]},
        }
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project='Wrist_GI_Original_Architecture')
wandb.agent(sweep_id, function=main)
