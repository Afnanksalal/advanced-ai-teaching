# GPU Lab Course: Sequence Models to Deployment

Hands-on labs with PDF study notes covering RNNs, attention, transformers, generative models, vision, reinforcement learning, Spark MLlib, MLOps, and explainability. Each module pairs a **PDF note** in `notes/` with a **fully implemented Jupyter notebook** in `labs/`.

## Requirements

- **Windows 10/11** (64-bit)
- **Python 3.10 or 3.11**
- **NVIDIA GPU** recommended (4 GB VRAM minimum; CPU fallback works for most labs)
- **16 GB RAM** (32 GB if running Spark + Jupyter together)
- **~15 GB disk** for the virtual environment and cached model weights

Optional for Lab 18 only: [Microsoft OpenJDK 17](https://learn.microsoft.com/en-us/java/openjdk/download)

## Quick start

```powershell
git clone https://github.com/Afnanksalal/advanced-ai-teaching.git
cd advanced-ai-teaching

# One-time setup (creates .venv, installs PyTorch + dependencies, registers Jupyter kernel)
setup.bat

# Every session: starts MLflow + Jupyter Lab
.\env.ps1
```

Open http://127.0.0.1:8888/lab, pick kernel **Python 3.11 (labs)**, and run notebooks from `labs/`.

Environment variables only (no servers):

```powershell
.\env.ps1 -EnvOnly
```

## Repository layout

```
advanced-ai-teaching/
├── labs/           # 20 Jupyter notebooks (complete code, not fill-in templates)
├── notes/          # 21 PDF study modules (00 = environment setup)
├── datasets/       # Small bundled samples; large sets download on first run
├── requirements.txt
├── setup.bat       # One-time Windows install
├── env.ps1         # Session launcher (venv + MLflow + Jupyter)
└── LICENSE
```

Not in this repo (local only): `.venv/`, `scripts/`, `tools/`, `hf_cache/`, `mlruns/`, downloaded MNIST/COCO weights.

## Modules

| # | PDF | Notebook |
|---|-----|----------|
| 00 | Environment setup | - |
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

Read the matching PDF before each lab. See `datasets/README.md` for what data ships vs downloads.

## CUDA check

After setup:

```powershell
.\env.ps1 -EnvOnly
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is `False`, reinstall PyTorch with CUDA wheels:

```powershell
.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Spark lab (Module 18)

Install JDK 17. PySpark on Windows also needs a Hadoop `winutils.exe` binary; see `notes/00-environment-setup.pdf` for details. `env.ps1` sets `JAVA_HOME` and `HADOOP_HOME` when those paths exist locally.

## MLflow (Module 19)

`env.ps1` starts the UI at http://127.0.0.1:5000. Runs are stored under `mlruns/` (gitignored).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `jupyter` not found | Run `setup.bat` or `.\env.ps1` (activates `.venv`) |
| CUDA OOM | Lower `batch_size` in the notebook |
| Missing package | `.venv\Scripts\activate` then `pip install -r requirements.txt` |
| Wrong kernel | Select **Python 3.11 (labs)** in Jupyter |

## License

[MIT](LICENSE). Copyright (c) 2026 Afnanksalal
