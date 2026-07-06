"""Download and prepare datasets for labs (tracked in git)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_env import configure_runtime, ensure_sample_images, load_sst2

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"


def ensure_dirs() -> None:
    for name in ("mnist", "coco128", "sample_images", "oxford_pet"):
        (DATASETS / name).mkdir(parents=True, exist_ok=True)


def download_mnist() -> None:
    print("MNIST, prefetching to datasets/mnist ...")
    dest = DATASETS / "mnist"
    if dest.exists() and any(dest.rglob("train-images-idx3-ubyte")):
        print("  OK MNIST (already on disk)")
        return
    try:
        import contextlib
        import io

        from torchvision import datasets

        with contextlib.redirect_stderr(io.StringIO()):
            datasets.MNIST(root=str(dest), train=True, download=True)
            datasets.MNIST(root=str(dest), train=False, download=True)
        print("  OK MNIST")
    except ImportError:
        print("  SKIP MNIST (install torch/torchvision first)")
    except Exception as exc:
        print(f"  WARN MNIST: {exc}")


def create_credit_fraud_sample(n_rows: int = 5000, seed: int = 42) -> None:
    print("Credit fraud sample...")
    out_csv = DATASETS / "credit_fraud_sample.csv"
    if out_csv.exists():
        print(f"  OK {out_csv} (already on disk)")
        return
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * 0.002)
    n_normal = n_rows - n_fraud
    rows = []
    for label, count in [(0, n_normal), (1, n_fraud)]:
        for _ in range(count):
            row = {"Time": rng.uniform(0, 172800), "Amount": rng.lognormal(3, 1.5)}
            for i in range(1, 29):
                row[f"V{i}"] = rng.normal(0, 1)
            row["Class"] = label
            rows.append(row)
    df = pd.DataFrame(rows)
    out_parquet = DATASETS / "credit_fraud_sample.parquet"
    try:
        df.to_parquet(out_parquet, index=False)
        print(f"  OK {out_parquet} ({len(df)} rows, {df['Class'].sum()} fraud)")
    except ImportError:
        df.to_csv(out_csv, index=False)
        print(f"  OK {out_csv} ({len(df)} rows, {df['Class'].sum()} fraud)")


def create_instruction_sample() -> None:
    print("Instruction sample for LoRA lab...")
    out = DATASETS / "instruction_sample.jsonl"
    if out.exists():
        print(f"  OK {out} (already on disk)")
        return
    samples = [
        {"instruction": "Summarize AI", "input": "", "output": "AI is the simulation of human intelligence by machines."},
        {"instruction": "Define machine learning", "input": "", "output": "ML learns patterns from data without explicit rules."},
        {"instruction": "Explain neural network", "input": "", "output": "Layers of connected units that transform inputs to outputs."},
        {"instruction": "What is deep learning?", "input": "", "output": "Deep learning uses many-layer neural networks."},
        {"instruction": "Describe supervised learning", "input": "", "output": "Learning from labeled input-output pairs."},
    ] * 100
    with out.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")
    print(f"  OK {out} ({len(samples)} rows)")


def prefetch_glue_sst2() -> None:
    print("SST-2 via HuggingFace datasets...")
    try:
        ds = load_sst2()
        print(f"  OK SST-2 train={len(ds['train'])} val={len(ds['validation'])}")
    except Exception as exc:
        print(f"  WARN SST-2: {exc}")


def prefetch_coco128() -> None:
    print("COCO128 / YOLO weights (optional prefetch)...")
    try:
        from ultralytics import YOLO

        YOLO("yolov8n.pt")
        print("  OK yolov8n weights cached")
    except ImportError:
        print("  SKIP (install ultralytics first)")
    except Exception as exc:
        print(f"  WARN: {exc}")


def download_oxford_pet() -> None:
    print("Oxford-IIIT Pet...")
    try:
        from torchvision.datasets import OxfordIIITPet

        OxfordIIITPet(root=str(DATASETS / "oxford_pet"), download=True)
        print("  OK Oxford-IIIT Pet")
    except ImportError:
        print("  SKIP Pet dataset (install torchvision first)")
    except Exception as exc:
        print(f"  WARN: {exc}")


def main() -> int:
    configure_runtime(quiet=True)
    parser = argparse.ArgumentParser(description="Prefetch datasets used by lab notebooks.")
    parser.add_argument("--with-pet", action="store_true", help="Download Oxford-IIIT Pet (~800MB)")
    parser.add_argument("--minimal", action="store_true", help="Only bundled samples + vision JPGs")
    args = parser.parse_args()

    ensure_dirs()
    create_credit_fraud_sample()
    create_instruction_sample()
    ensure_sample_images()
    print("  OK sample_images")

    if args.minimal:
        print("\nDone (minimal). Large sets download on first lab run.")
        return 0

    download_mnist()
    prefetch_glue_sst2()
    prefetch_coco128()
    if args.with_pet:
        download_oxford_pet()
    print("\nDone. See datasets/README.md for details.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
