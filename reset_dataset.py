import os
import shutil
import json
from src.train import prepare_mlx_data

os.makedirs("data/archived_rounds", exist_ok=True)

# Move augmented jsonl files
for i in range(1, 7):
    src = f"data/train_round_{i}.jsonl"
    if os.path.exists(src):
        os.rename(src, f"data/archived_rounds/train_round_{i}.jsonl")
        
# Clean data_mlx
if os.path.exists("data_mlx"):
    shutil.rmtree("data_mlx")
os.makedirs("data_mlx", exist_ok=True)

prepare_mlx_data("data/train.jsonl", "data_mlx/temp.jsonl")

# Fix directory structure from prepare_mlx_data bug
os.rename("data_mlx/temp.jsonl/train.jsonl", "data_mlx/train.jsonl")
os.rmdir("data_mlx/temp.jsonl")

print("Reset complete! Original dataset data/train.jsonl is prepared in data_mlx/train.jsonl")
