# Datasets for the Advanced AI course

This folder holds data files referenced by the lab notebooks. Some files are committed to git; larger sets download automatically the first time you run the relevant lab.

## Included in the repository

| File / folder | Labs | Description |
|---------------|------|-------------|
| `credit_fraud_sample.csv` | 18, 20 | Synthetic credit-card fraud features (V1 to V28, Amount, Class) |
| `instruction_sample.jsonl` | 11 | Instruction tuning examples for LoRA fine-tuning of GPT-2 |
| `sample_images/` | 12, 14 | JPG images for object detection and segmentation inference |

If `sample_images/` is empty, run `python download_datasets.py --minimal` from the repo root. The script downloads default images from public URLs used in the Ultralytics documentation.

## Downloaded on first lab run

| Data | Labs | Source |
|------|------|--------|
| MNIST | 05, 06, 07, 08, 19 | `torchvision.datasets.MNIST` into `datasets/mnist/` |
| SST-2 sentiment | 09 | Hugging Face `SetFit/sst2` via `datasets` |
| DistilBERT / GPT-2 weights | 09, 10, 11 | Hugging Face Hub, cached under `hf_cache/` |
| COCO128 | 13 | Ultralytics dataset bundle on first `model.train()` |
| YOLOv8n weights | 13 | Ultralytics on first `YOLO("yolov8n.pt")` |

Run your session with `env.ps1` or `env.sh` so Hugging Face and MLflow caches stay inside the repo directory rather than your user profile.

## Prefetch commands

Run from the repository root with `.venv` activated (or after `env.ps1` / `env.sh`):

```bash
python download_datasets.py --minimal
```

Downloads bundled sample images and verifies CSV/JSONL files. This is what `setup.bat` and `setup.sh` call during install.

```bash
python download_datasets.py
```

Additionally prefetches MNIST, SST-2, and YOLOv8n weights so later labs start faster.

```bash
python download_datasets.py --with-pet
```

Optional: downloads the Oxford-IIIT Pet dataset (~800 MB) for extended segmentation experiments. Not required for the core twenty modules.
