import json
import os
from src.train import prepare_mlx_data

base_file = "data/train_round_5.jsonl"
output_file = "data/train_round_6.jsonl"
mlx_dir = "data_mlx"

# Read base dataset
dataset = []
with open(base_file, "r") as f:
    for line in f:
        if line.strip():
            dataset.append(json.loads(line))

# Our exact failure we want to fix
failure_input = "What is 2+2?"
perfect_response = "4"

# Adapt data augmentation strategy: append 15 copies to increase weight
new_data = {"messages": [{"role": "user", "content": failure_input}, {"role": "assistant", "content": perfect_response}]}
for _ in range(15):
    dataset.append(new_data)

# Write to output file
with open(output_file, "w") as f:
    for ex in dataset:
        f.write(json.dumps(ex) + "\n")

# Prepare MLX data
# Remove old data_mlx file to avoid FileExistsError
if os.path.exists("data_mlx/train.jsonl"):
    if os.path.isdir("data_mlx/train.jsonl"):
        import shutil
        shutil.rmtree("data_mlx/train.jsonl")
    else:
        os.remove("data_mlx/train.jsonl")

os.makedirs(mlx_dir, exist_ok=True)
prepare_mlx_data(output_file, os.path.join(mlx_dir, "train.jsonl"))

# Fix directory bug immediately
os.rename("data_mlx/train.jsonl/train.jsonl", "data_mlx/temp.jsonl")
os.rmdir("data_mlx/train.jsonl")
os.rename("data_mlx/temp.jsonl", "data_mlx/train.jsonl")

print(f"Data successfully augmented with 15 copies and prepared at data_mlx/train.jsonl")
