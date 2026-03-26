# Multimodal Foundation Models Framework

This repository implements a modular framework for training, evaluating, and benchmarking multimodal foundation models. It specifically supports **CLIP** (Contrastive Language-Image Pre-training) baselines, **CLIP with LoRA** (Low-Rank Adaptation), and **Frozen** architectures (combining frozen vision encoders with language models).

The project is designed for research purposes, facilitating the comparison of full fine-tuning vs. parameter-efficient fine-tuning (PEFT) and benchmarking efficiency metrics (latency, memory, parameters) suitable for robotics and edge applications.

## 📂 Project Structure

```text
├── benchmark/           # Benchmarking scripts for latency/memory profiling
├── configs/             # YAML configuration files for different models
├── datasets/            # Dataset loading and processing logic
│   ├── conceptual_captions.py  # Training dataset implementation
│   └── benchmark_datasets.py   # Eval datasets (CIFAR, Food101, etc.)
├── evaluation/          # Evaluation metrics, profiling, and caching
├── models/              # Model definitions (CLIP, LoRA, Frozen)
├── notebooks/           # Jupyter notebooks for experiments and testing
├── training/            # Training loops and trainer classes
├── utils/               # Helpers, config, transforms, and model merging logic
├── visual_testing/      # Visual sanity checks and notebooks
├── main.py              # Main CLI entry point for the project
└── requirements.txt     # (Implicit) Dependencies
🛠️ Installation
Ensure you have Python 3.8+ and the following dependencies installed:

Bash

pip install torch torchvision
pip install transformers peft
pip install pyyaml tqdm scikit-learn
pip install matplotlib seaborn
pip install ipykernel  # For notebooks
⚙️ Configuration
The system uses YAML files located in configs/ to manage hyperparameters.

clip_baseline.yaml: Standard CLIP fine-tuning or evaluation.

clip_lora.yaml: Configuration for PEFT using LoRA (Low-Rank Adaptation). Includes settings for lora_r, lora_alpha, and target modules.

frozen_clip.yaml: Configuration for the "Frozen" architecture (e.g., ResNet + GPT2).

Example Config (configs/clip_lora.yaml):

YAML

model:
  model_id: "openai/clip-vit-base-patch32"
  lora_r: 16
  target_modules: ["q_proj", "v_proj"]

training:
  batch_size: 64
  num_epochs: 3
  learning_rate: 5.0e-5
  output_dir: "clip_lora_checkpoints"
🚀 Usage
The primary entry point is main.py. It supports three modes: train, evaluate, and benchmark.

1. Training
Train a model using the settings defined in your config file.

Train CLIP with LoRA:

Bash

python main.py --mode train --model clip_lora --config configs/clip_lora.yaml --gpu 0
Train Frozen Model:

Bash

python main.py --mode train --model frozen --config configs/frozen_clip.yaml --gpu 0
2. Evaluation
Evaluate a trained model on downstream classification datasets (Zero-shot / Few-shot).

Bash

python main.py --mode evaluate --model clip_lora --datasets cifar100 food101 --gpu 0
Supported Datasets: cifar100, food101, flowers102, dtd, eurosat.

Note: For clip_lora, the evaluator automatically detects and loads the latest adapter checkpoint or merged model.

3. Benchmarking
Run efficiency benchmarks (Parameter counts, VRAM usage, Latency).

Option A: Using the Main CLI Runs the standard profiling suite integrated into the main application.

Bash

python main.py --mode benchmark --model clip_lora
Option B: Using the Dedicated Script (Detailed Reporting) This script generates a JSON report and formatted LaTeX tables for publication.

⚠️ Important: Before running this script, you must update the CHECKPOINT_PATHS dictionary in benchmark/run_benchmark.py to point to your actual local checkpoint files.

Bash

python benchmark/run_benchmark.py --device cuda
Features:

Compares Merged vs. Unmerged LoRA adapters.

Measures Vision Latency vs. End-to-End Generation Latency.

Strict FP16 precision setup for fair comparison.

🧠 Supported Models
CLIP Baseline: Standard implementation using Hugging Face's CLIPModel.

CLIP + LoRA: Applies Low-Rank Adaptation to specific projection layers (q_proj, v_proj) of the CLIP model, allowing for efficient fine-tuning.

Frozen: A composite model typically using a frozen Vision Encoder (e.g., ResNet50) and a Large Language Model (e.g., GPT-2), focusing on captioning or generative tasks.

📊 Datasets
Training: The framework is configured to use the Conceptual Captions dataset. Ensure your data is organized as JSONL files (see configs/clip_lora.yaml for expected paths).

Evaluation: Standard computer vision benchmarks (CIFAR-100, EuroSAT, DTD, etc.) are implemented in datasets/benchmark_datasets.py.

📝 License & Notes
Source: Developed as part of a research project on Multimodal Foundation Models.