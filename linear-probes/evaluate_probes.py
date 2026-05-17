# EVALUATE ONLY AMONGUS / ROLEPLAYING LINEAR PROBE

import os
import sys
import json
import pickle
from typing import Dict, Any

import torch as t
import pandas as pd
from tqdm import tqdm

sys.path.append(".")
sys.path.append(os.path.dirname(os.path.abspath(".")))

from probe_datasets import RolePlayingDataset
from evaluate_utils import evaluate_probe_on_activation_dataset
from configs import config_qwen
from plots import plot_roc_curve_eval
from probes import LinearProbe


config = config_qwen
model, tokenizer, device = None, None, "cpu"


def evaluate_probe(
    dataset_name: str,
    probe: LinearProbe,
    config: Dict[str, Any],
    model=None,
    tokenizer=None,
) -> Dict[str, Any]:

    results_dir = f"results/{dataset_name}_{config['short_name']}"
    os.makedirs(results_dir, exist_ok=True)

    dataset = RolePlayingDataset(
        config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        test_split=0.2,
    )

    test_acts_chunk = dataset.get_test_acts(chunk_idx=0)

    probe_outputs, accuracy = evaluate_probe_on_activation_dataset(
        chunk_data=test_acts_chunk,
        probe=probe,
        device=device,
        num_tokens=None,
        verbose=False,
    )

    labels = t.tensor([batch[1] for batch in test_acts_chunk]).numpy()

    fig, roc = plot_roc_curve_eval(
        labels=labels,
        probe_outputs=probe_outputs,
    )

    fig.write_image(f"{results_dir}/roc_{dataset_name}.pdf", scale=1)

    metrics = {
        "dataset": dataset_name,
        "accuracy": float(accuracy),
        "roc": roc,
    }

    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {accuracy}")
    print(f"Saved ROC plot to {results_dir}/roc_{dataset_name}.pdf")
    print(f"Saved metrics to {results_dir}/metrics.json")

    return metrics


if __name__ == "__main__":

    dataset_name = "RolePlayingDataset"

    checkpoint_path = f"checkpoints/{dataset_name}_probe_{config['short_name']}.pkl"

    print(f"Loading probe from {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        probe = pickle.load(f)

    print("Loaded probe.")

    metrics = evaluate_probe(
        dataset_name=dataset_name,
        probe=probe,
        config=config,
        model=model,
        tokenizer=tokenizer,
    )

    print("Evaluation complete.")