# Datasets

## Included in the repository

| File / folder | Used in |
|---------------|---------|
| `credit_fraud_sample.csv` | Lab 18 (Spark), Lab 20 (XAI) |
| `instruction_sample.jsonl` | Lab 11 (LoRA) |
| `sample_images/` | Lab 12 (Faster R-CNN), Lab 14 (segmentation) |

## Downloaded automatically on first use

| Data | Lab | How |
|------|-----|-----|
| MNIST | 05 to 08, 19 | `torchvision.datasets.MNIST` |
| SST-2 | 09 | Hugging Face `SetFit/sst2` |
| GPT-2 / DistilBERT weights | 09 to 11 | Hugging Face Hub → `hf_cache/` |
| COCO128 | 13 | Ultralytics on first `model.train()` |
| YOLOv8n weights | 13 | Ultralytics on first `YOLO(...)` |

Keep `env.ps1` / `env.sh` loaded so Hugging Face and MLflow caches land in the project folder.

## Optional prefetch

```bash
python download_datasets.py          # bundled samples + MNIST + SST-2 + YOLO weights
python download_datasets.py --minimal  # only CSV/JSONL + vision sample JPGs (setup runs this)
```
