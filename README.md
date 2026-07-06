# Advanced AI

Teaching materials for a 20-module advanced AI course: **PDF study notes** and **complete Jupyter labs** from sequence models through deployment and explainability.

Each module pairs one PDF in `notes/` with one runnable notebook in `labs/`. Notebooks ship with full code (no fill-in sections), explanatory markdown, and automatic CPU/GPU detection.

## What you get

| Track | Modules | Topics |
|-------|---------|--------|
| Sequence models | 01–04 | RNNs, LSTM/GRU, attention, transformers |
| Generative AI | 05–08 | GANs, DCGAN, diffusion, DDPM |
| Language models | 09–11 | BERT, GPT, LoRA / PEFT |
| Computer vision | 12–14 | Faster R-CNN, YOLO, segmentation |
| Reinforcement learning | 15–17 | Q-learning, DQN, actor-critic |
| Scale & production | 18–20 | Spark MLlib, DDP & MLflow, SHAP/LIME & Flask |

Module 00 (`notes/00-environment-setup.pdf`) covers installation, CUDA, Spark, and troubleshooting.

## Requirements

- **Python 3.10 or 3.11**
- **Windows 10/11**, **Linux**, or **macOS**
- **16 GB RAM** (32 GB if running Spark and Jupyter together)
- **~15 GB disk** for the virtual environment and cached model weights
- **NVIDIA GPU** recommended (4 GB VRAM minimum); most labs run on CPU with longer training times

Optional for Module 18 only: **Java 17** ([Microsoft OpenJDK](https://learn.microsoft.com/en-us/java/openjdk/download) on Windows; OpenJDK 17 package on Linux)

## Quick start

**Windows**

```powershell
git clone https://github.com/Afnanksalal/advanced-ai-teaching.git
cd advanced-ai-teaching
setup.bat
.\env.ps1
```

**Linux / macOS**

```bash
git clone https://github.com/Afnanksalal/advanced-ai-teaching.git
cd advanced-ai-teaching
chmod +x setup.sh env.sh
./setup.sh
./env.sh
```

Open http://127.0.0.1:8888/lab, select kernel **Python 3.11 (labs)**, and run notebooks from `labs/`.

Load environment variables without starting servers:

```powershell
.\env.ps1 -EnvOnly          # Windows
source ./env.sh --env-only  # Linux/macOS
```

## Repository layout

```
advanced-ai-teaching/
├── labs/                 # 20 Jupyter notebooks (complete implementations)
├── notes/                # 21 PDF modules (00 = environment setup)
├── datasets/             # Bundled samples; large sets download on first run
├── runtime_env.py        # Portable bootstrap (HF cache, MLflow, Spark helpers)
├── download_datasets.py  # Prefetch bundled and optional large datasets
├── configure_kernel.py   # Registers the Jupyter kernel for all labs
├── requirements.txt
├── setup.bat / setup.sh  # One-time install
├── env.ps1 / env.sh      # Session launcher (venv + MLflow + Jupyter)
└── LICENSE
```

Local-only (created at runtime, not in git): `.venv/`, `hf_cache/`, `mlruns/`, `models/`, downloaded MNIST/COCO/HF weights.

## Modules

| # | PDF | Notebook |
|---|-----|----------|
| 00 | Environment setup | — |
| 01 | RNNs & sequence modeling | `lab01_rnn_sequence_modeling.ipynb` |
| 02 | LSTM & GRU | `lab02_lstm_gru_comparison.ipynb` |
| 03 | Attention & seq2seq | `lab03_attention_seq2seq.ipynb` |
| 04 | Transformers | `lab04_transformer_block.ipynb` |
| 05 | GAN fundamentals | `lab05_gan_mnist.ipynb` |
| 06 | DCGAN | `lab06_dcgan_mnist.ipynb` |
| 07 | Diffusion theory | `lab07_diffusion_visualization.ipynb` |
| 08 | DDPM | `lab08_ddpm_mnist.ipynb` |
| 09 | BERT | `lab09_bert_finetune.ipynb` |
| 10 | GPT / causal LMs | `lab10_gpt_generation.ipynb` |
| 11 | LoRA / PEFT | `lab11_lora_finetune.ipynb` |
| 12 | Faster R-CNN | `lab12_faster_rcnn_inference.ipynb` |
| 13 | YOLO & mAP | `lab13_yolo_training.ipynb` |
| 14 | Segmentation | `lab14_segmentation.ipynb` |
| 15 | Q-learning | `lab15_qlearning_frozenlake.ipynb` |
| 16 | DQN | `lab16_dqn_cartpole.ipynb` |
| 17 | Actor-critic | `lab17_a2c_cartpole.ipynb` |
| 18 | Spark MLlib | `lab18_spark_mllib.ipynb` |
| 19 | DDP & MLflow | `lab19_ddp_mlflow.ipynb` |
| 20 | SHAP, LIME, deployment | `lab20_xai_deployment.ipynb` |

Read the matching PDF before each lab. See `datasets/README.md` for bundled vs auto-downloaded data.

## CUDA check

After setup:

```powershell
.\env.ps1 -EnvOnly
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is `False` on a machine with an NVIDIA GPU, reinstall PyTorch with CUDA wheels:

```powershell
.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

On Linux/macOS without a GPU, `setup.sh` installs the CPU build automatically.

## Spark (Module 18)

Install JDK 17. On Windows, PySpark also needs a Hadoop `winutils.exe` in `tools/hadoop/bin/` (see `notes/00-environment-setup.pdf`). Session launchers set `JAVA_HOME` when a JDK is found.

## MLflow (Module 19)

`env.ps1` / `env.sh` start the UI at http://127.0.0.1:5050. Runs are stored in `mlruns/mlflow.db` (SQLite, gitignored).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `jupyter` not found | Run `setup.bat` / `./setup.sh`, then `env.ps1` / `env.sh` |
| CUDA OOM | Lower `batch_size` in the notebook |
| Missing package | Activate `.venv`, then `pip install -r requirements.txt` |
| Wrong kernel | Select **Python 3.11 (labs)**; re-run `python configure_kernel.py` |
| Missing sample images | `python download_datasets.py --minimal` |

## License

[MIT](LICENSE). Copyright (c) 2026 Afnanksalal
