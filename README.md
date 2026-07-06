# Advanced AI

This repository is the complete **Advanced AI** course you work through as a student. It contains twenty graded modules plus a setup guide. For each module you read one PDF in `notes/` and complete one Jupyter notebook in `labs/`. The notebooks are fully implemented: every code cell runs as written, with markdown explaining the models, loss functions, and training steps. You are not expected to fill in skeleton code or locate supplementary handouts elsewhere.

The course moves from recurrent sequence models through transformers, generative models (GANs and diffusion), NLP fine-tuning, computer vision, reinforcement learning, distributed tabular ML with Spark, multi-GPU training with MLflow, and model explainability with deployment.

## How to work through the course

1. Clone this repository and run the one-time setup script for your operating system (`setup.bat` on Windows, `setup.sh` on Linux or macOS).
2. Read `notes/00-environment-setup.pdf` and confirm Python, Jupyter, and (if available) CUDA are working.
3. For each module `N` from 01 to 20:
   - Read `notes/NN-*.pdf` for theory, equations, and references.
   - Open the matching notebook in `labs/` and run all cells top to bottom.
4. Start each study session with `env.ps1` (Windows) or `env.sh` (Linux/macOS) so the virtual environment, cache paths, MLflow, and Jupyter Lab are configured consistently.

Module PDFs cross-reference each other by module number (for example, Module 07 before Module 08). Follow that order unless your instructor specifies otherwise.

## System requirements

| Component | Requirement |
|-----------|-------------|
| Operating system | Windows 10/11, Linux, or macOS (64-bit) |
| Python | 3.10 or 3.11 |
| RAM | 16 GB minimum; 32 GB recommended if you run Spark and Jupyter together |
| Disk | About 15 GB free for the virtual environment, cached Hugging Face weights, and downloaded datasets |
| GPU | NVIDIA GPU with 4 GB VRAM or more recommended; CPU-only execution is supported but training will take longer |
| Network | Required on first run for several labs (Hugging Face models, MNIST, COCO128, YOLO weights, sample images) |

**Module 18 only:** Java 17 (OpenJDK). On Windows, PySpark also expects a Hadoop `winutils.exe` binary under `tools/hadoop/bin/`. See `notes/00-environment-setup.pdf` for installation steps.

## Installation

### Windows

```powershell
git clone https://github.com/Afnanksalal/advanced-ai-teaching.git
cd advanced-ai-teaching
setup.bat
```

`setup.bat` creates `.venv`, installs PyTorch (CUDA 12.1 wheels when available), installs `requirements.txt`, registers the **Python 3.11 (labs)** Jupyter kernel via `configure_kernel.py`, and prefetches bundled datasets with `download_datasets.py --minimal`.

### Linux and macOS

```bash
git clone https://github.com/Afnanksalal/advanced-ai-teaching.git
cd advanced-ai-teaching
chmod +x setup.sh env.sh
./setup.sh
```

On machines without an NVIDIA GPU, `setup.sh` installs the CPU build of PyTorch automatically.

### Every session

```powershell
# Windows (PowerShell, from the repo root)
.\env.ps1
```

```bash
# Linux / macOS
./env.sh
```

This activates `.venv`, sets Hugging Face and MLflow cache directories under the repo, starts MLflow UI on port 5050, and launches Jupyter Lab on port 8888. Select kernel **Python 3.11 (labs)**.

To load environment variables without starting servers:

```powershell
.\env.ps1 -EnvOnly          # Windows
source ./env.sh --env-only  # Linux / macOS
```

## Repository layout

```
advanced-ai-teaching/
├── notes/                  Module PDFs (00 = environment and tooling)
├── labs/                   Twenty lab notebooks (lab01 through lab20)
├── datasets/               Bundled CSV, JSONL, and sample images
├── runtime_env.py            Sets HF_HOME, MLFLOW_TRACKING_URI, Java/Spark paths
├── configure_kernel.py       Installs the Jupyter kernel with course env vars
├── download_datasets.py      Prefetch MNIST, SST-2, sample images, and related data
├── requirements.txt          Pinned Python dependencies for all modules
├── setup.bat / setup.sh      One-time install (run once after clone)
├── env.ps1 / env.sh          Session launcher (run each time you study)
└── LICENSE
```

The following directories are created locally and are not part of the git repository: `.venv/` (Python environment), `hf_cache/` (Hugging Face downloads), `mlruns/` (MLflow SQLite store), `models/` (saved lab artifacts such as the fraud classifier in Module 20), and Ultralytics `runs/` output from Module 13.

## Module list

| Module | PDF topic | Lab notebook | Main libraries / datasets |
|--------|-----------|--------------|---------------------------|
| 00 | Environment setup | (setup guide only) | Python, Jupyter, CUDA, Spark, MLflow |
| 01 | RNNs and sequence modeling | `lab01_rnn_sequence_modeling.ipynb` | PyTorch, character-level LM |
| 02 | LSTM and GRU | `lab02_lstm_gru_comparison.ipynb` | PyTorch, sequential classification |
| 03 | Attention and seq2seq | `lab03_attention_seq2seq.ipynb` | PyTorch, additive attention |
| 04 | Transformers | `lab04_transformer_block.ipynb` | PyTorch, multi-head self-attention |
| 05 | GAN fundamentals | `lab05_gan_mnist.ipynb` | PyTorch, MNIST |
| 06 | DCGAN | `lab06_dcgan_mnist.ipynb` | PyTorch, transposed convolutions, MNIST |
| 07 | Diffusion theory | `lab07_diffusion_visualization.ipynb` | PyTorch, forward noise schedule |
| 08 | DDPM | `lab08_ddpm_mnist.ipynb` | PyTorch, U-Net denoiser, MNIST |
| 09 | BERT | `lab09_bert_finetune.ipynb` | Hugging Face Transformers, SST-2 |
| 10 | GPT / causal LMs | `lab10_gpt_generation.ipynb` | Hugging Face, decoding strategies |
| 11 | LoRA / PEFT | `lab11_lora_finetune.ipynb` | PEFT, GPT-2, instruction JSONL |
| 12 | Faster R-CNN | `lab12_faster_rcnn_inference.ipynb` | torchvision detection, sample images |
| 13 | YOLO and mAP | `lab13_yolo_training.ipynb` | Ultralytics, COCO128 |
| 14 | Segmentation | `lab14_segmentation.ipynb` | DeepLabV3, sample images |
| 15 | Q-learning | `lab15_qlearning_frozenlake.ipynb` | Gymnasium, tabular RL |
| 16 | DQN | `lab16_dqn_cartpole.ipynb` | Stable-Baselines3, CartPole |
| 17 | Actor-critic | `lab17_a2c_cartpole.ipynb` | Stable-Baselines3, A2C |
| 18 | Spark MLlib | `lab18_spark_mllib.ipynb` | PySpark, credit fraud sample |
| 19 | DDP and MLflow | `lab19_ddp_mlflow.ipynb` | PyTorch, MLflow, MNIST |
| 20 | SHAP, LIME, deployment | `lab20_xai_deployment.ipynb` | scikit-learn, SHAP, LIME, Flask |

See `datasets/README.md` for which files ship in git versus which download on first lab execution.

## Course tracks (summary)

**Modules 01 to 04 (sequence models):** Vanishing gradients in vanilla RNNs, LSTM/GRU gating, Bahdanau attention, scaled dot-product self-attention and transformer blocks.

**Modules 05 to 08 (generative models):** GAN minimax training, DCGAN architectural rules, forward/reverse diffusion, DDPM noise prediction and sampling.

**Modules 09 to 11 (language models):** BERT masked LM fine-tuning, GPT causal generation and decoding, LoRA parameter-efficient adaptation.

**Modules 12 to 14 (vision):** Two-stage detection with Faster R-CNN, single-shot YOLO training and mAP, semantic segmentation with DeepLabV3.

**Modules 15 to 17 (reinforcement learning):** MDPs and Q-learning, deep Q-networks with replay, actor-critic policy gradients.

**Modules 18 to 20 (scale and production):** Distributed logistic regression and random forests in Spark MLlib, experiment tracking with MLflow, SHAP/LIME attributions and a Flask inference endpoint.

## Verify CUDA

After setup:

```powershell
.\env.ps1 -EnvOnly
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If you have an NVIDIA GPU but CUDA reports `False`, reinstall PyTorch with CUDA wheels:

```powershell
.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Reduce `batch_size` in the notebook if you hit CUDA out-of-memory errors. Modules 09 to 11 document conservative batch sizes for 4 GB cards.

## Spark (Module 18)

Install JDK 17 and ensure `JAVA_HOME` points to it. On Windows, place `winutils.exe` in `tools/hadoop/bin/` as described in Module 00. The session scripts set `JAVA_HOME` and `HADOOP_HOME` when those paths exist.

## MLflow (Module 19)

`env.ps1` and `env.sh` start the MLflow UI at http://127.0.0.1:5050. Experiment data is stored in `mlruns/mlflow.db` (SQLite). Lab 19 logs hyperparameters, per-epoch loss, test accuracy, and a saved PyTorch model artifact to this store.

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `jupyter: command not found` | Run `setup.bat` or `./setup.sh`, then start a session with `env.ps1` or `env.sh` |
| Wrong Jupyter kernel | Kernel menu, change to **Python 3.11 (labs)**; or run `python configure_kernel.py` |
| Missing Python package | Activate `.venv`, then `pip install -r requirements.txt` |
| Missing sample images (Labs 12, 14) | `python download_datasets.py --minimal` |
| Hugging Face download errors | Ensure `env.ps1` or `env.sh` was loaded; check network access |
| MLflow UI not reachable | Confirm port 5050 is free; see `.logs/mlflow-ui.log` after running `env.ps1` |
| Spark / winutils errors (Windows) | Follow Module 00; restart terminal after setting `JAVA_HOME` |

## License

[MIT](LICENSE). Copyright (c) 2026 Afnanksalal
