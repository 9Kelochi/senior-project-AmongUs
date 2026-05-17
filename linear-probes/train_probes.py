import os
import sys
import pickle
from typing import List
from configs import config_qwen

sys.path.append(os.path.dirname(os.path.abspath('.')))
sys.path.append('.')

from probe_datasets import RolePlayingDataset
from probes import LinearProbe

# Training RolePlayingDataset with layer 14
datasets: List[str] = ["RolePlayingDataset"]

config = config_qwen
model, tokenizer, device = None, None, 'cpu'

################### TRAINING PROBES

for dataset_name in datasets:
    print(f"Loading {dataset_name}...")

    dataset = RolePlayingDataset(
        config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        test_split=0.2,
    )

    train_loader = dataset.get_train(
        batch_size=config["probe_training_batch_size"],
        num_tokens=config["probe_training_num_tokens"],
        chunk_idx=config["probe_training_chunk_idx"],
    )

    sample_X, sample_y = next(iter(train_loader))
    actual_input_dim = sample_X.shape[-1]

    print(f"Detected activation size: {actual_input_dim}")

    probe = LinearProbe(
        input_dim=actual_input_dim,
        device=device,
        lr=config["probe_training_learning_rate"]
    )

    print(f'Training probe on {len(train_loader)} batches and {len(train_loader.dataset)} samples.')
    probe.fit(train_loader, epochs=config["probe_training_epochs"])

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{dataset_name}_probe_{config['short_name']}.pkl"
    )

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(probe, f)

    print(f"Saved to {checkpoint_path}")

print("Probe training complete for RolePlayingDataset.")