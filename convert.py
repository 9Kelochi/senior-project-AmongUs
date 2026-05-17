import json
import pandas as pd
from pathlib import Path

LOGS_ROOT = Path("expt-logs")
INPUT_FOLDER = "2026-05-16_exp_0"
OUTPUT_CSV = Path("linear-probes/data/Roleplaying/phi4_rollouts.csv")

REQUIRED_COLUMNS = [
    "scenario",
    "question",
    "answer_prefix",
    "phi4_rollout",
    "llm_eval_gt",
]


def read_json_objects(path: Path):
    text = path.read_text(encoding="utf-8")

    decoder = json.JSONDecoder()
    idx = 0
    entries = []

    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1

        if idx >= len(text):
            break

        try:
            obj, end = decoder.raw_decode(text, idx)
            entries.append(obj)
            idx = end
        except json.JSONDecodeError:
            print(f"JSON parse stopped in {path} at:", idx)
            print(text[idx:idx + 200])
            break

    return entries


rows = []
total_entries = 0
used_files = 0

folder = LOGS_ROOT / INPUT_FOLDER

if not folder.exists():
    raise FileNotFoundError(f"Could not find folder: {folder}")

if not folder.is_dir():
    raise NotADirectoryError(f"This is not a folder: {folder}")

print(f"Using folder: {folder}")

log_path = folder / "agent-logs.json"

if not log_path.exists():
    raise FileNotFoundError(f"Could not find log file: {log_path}")

print(f"Reading {log_path}")
entries = read_json_objects(log_path)

total_entries += len(entries)
used_files += 1

for data in entries:
    interaction = data.get("interaction", {})
    prompt = interaction.get("prompt", {})

    scenario = str(prompt.get("Summarization", "") or "")

    question = (
        f"Summarization: {prompt.get('Summarization', '') or ''}\n\n"
        f"{prompt.get('All Info', '') or ''}\n\n"
        f"Memory: {prompt.get('Memory', '') or ''}\n\n"
        f"Phase: {prompt.get('Phase', '') or ''}. Return your output."
    )

    identity = data.get("player", {}).get("identity", "")
    llm_eval_gt = 1 if identity == "Impostor" else 0

    rollout = str(interaction.get("full_response", "") or "")

    action_index = rollout.find("[Action]")

    if action_index != -1:
        answer_prefix = rollout[:action_index].strip()
        phi4_rollout = rollout[action_index:].strip()
    else:
        answer_prefix = ""
        phi4_rollout = rollout.strip()

    if not question.strip() or not phi4_rollout.strip():
        continue

    rows.append({
        "scenario": scenario,
        "question": question,
        "answer_prefix": answer_prefix,
        "phi4_rollout": phi4_rollout,
        "llm_eval_gt": llm_eval_gt,
    })


df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)

df["scenario"] = df["scenario"].fillna("").astype(str)
df["question"] = df["question"].fillna("").astype(str)
df["answer_prefix"] = df["answer_prefix"].fillna("").astype(str)
df["phi4_rollout"] = df["phi4_rollout"].fillna("").astype(str)
df["llm_eval_gt"] = df["llm_eval_gt"].fillna(0).astype(int)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, na_rep="")

print(f"\nUsed {used_files} log files")
print(f"Parsed {total_entries} JSON objects")
print(f"Saved {len(df)} rows to {OUTPUT_CSV}")

print("\nLabel distribution:")
print(df["llm_eval_gt"].value_counts())

print("\nColumn types:")
print(df.dtypes)