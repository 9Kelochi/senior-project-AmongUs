import os

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_path = f'{checkpoint_dir}/{dataset_name}_probe_{config["short_name"]}.pkl'

with open(checkpoint_path, 'wb') as f:
    pickle.dump(probe, f)