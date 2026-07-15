"""scripts/01_download.py — Direct parquet download."""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

# 国内用户取消注释
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "swe_rebench": "nebius/SWE-rebench-openhands-trajectories",
    "swe_gym_sampled": "SWE-Gym/OpenHands-Sampled-Trajectories",
    "swe_bench_lite": "SWE-bench/SWE-bench_Lite",
    "swe_bench_verified": "SWE-bench/SWE-bench_Verified",
}


def download_all():
    for name, repo_id in DATASETS.items():
        target = RAW_DIR / name
        if target.exists() and any(target.iterdir()):
            print(f"[skip] {name} already downloaded")
            continue
        print(f"[download] {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=["*.parquet", "*.json", "README.md"],
        )
        files = list(target.rglob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files) / 1e9
        print(f"  → {len(files)} parquet files, {total_size:.2f} GB")


if __name__ == "__main__":
    download_all()