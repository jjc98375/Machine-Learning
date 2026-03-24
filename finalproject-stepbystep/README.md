# Predictive Code-Switching in Bilingual Discourse

## Project Overview
This repository contains the step-by-step implementation of a predictive machine learning pipeline designed to anticipate **Code-Switching** boundaries (changes in language) and their consecutive durations within bilingual text. Operating on the underlying structures of large pre-trained multilingual transformers like **XLM-RoBERTa** and **mBERT**, the system learns cross-lingual patterns to predict precisely when a speaker/writer will switch languages.

## Final Project Guideline (Step-by-Step)
To ensure a robust and debuggable ML lifecycle, the project has been carefully isolated into distinct, incremental stages: from streaming massive datasets to fully training dual-head architectures.

---

### 🟢 Phase 1: Data Pipeline & Baseline Architecture
The first phase establishes the underlying infrastructure. It correctly streams, cleans, and standardizes multi-lingual text to prepare for neural network training.
* **Streaming Dataset Loader (`dataset.py`)**: Securely fetches the massive gated `SwitchLingua` datasets in real-time via the Hugging Face hub, thereby circumventing local storage limits.
* **Tokenization & Masking**: Correctly parses token arrays, aligns subwords to labels, and implements causal masks to hide futuristic context.
* **Baseline Calculation (`baseline.py`)**: Creates naive prediction pipelines and outlines the strict evaluation math (`Anticipatory F1 Score`, `Macro F1`) that forms the standard benchmarks.

---

### 🔵 Phase 2: Deep Learning Training & Evaluation
The second phase handles actual intelligence via Transformer models, executing experiments, computing loss metrics, and capturing final evaluation graphics.
* **Dual-Head Neural Network (`model.py`)**: Builds two custom classification heads on the transformer outputs:
  1. `Switch Head` (BCE Loss) for detecting language change.
  2. `Duration Head` (CrossEntropy Loss) to gauge the span of the code-switch. 
  *(Includes dynamic token filters to prevent `-100` padding calculation overflows).*
* **Training Engine (`train.py`)**: Implements the AdamW optimizer with linear warmup, handling the Epoch loops, learning rates, checkpoint creations, and auto-detecting hardware (CUDA/MPS).
* **Automated Experimentation (`run_experiment.py`)**: A centralized CLI controller used to run both backbones (XLM-R vs. mBERT) simultaneously. Generates a robust summary output (`experiment_history.txt`) cataloging all hyperparameters, per-pair F1 results, and duration accuracies instantly.
* **Visualization Studio (`visualize.py`)**: Translates numeric dictionaries into comprehensible Matplotlib graphs. Outputs line graphs representing training convergence (`total_loss_comparison.png`) and bar charts outlining macro F1 metrics.

---

## How to Run
Navigate to the `phase2` directory and execute the foundational testing script:

```bash
# Run a quick smoke test
python run_experiment.py --epochs 1 --samples_per_pair 200

# Execute the complete, full-scale final training pipeline
python run_experiment.py --epochs 5 --samples_per_pair 2000
```
All outputs (model neural weights, graphical plots, and experimental text histories) will be automatically preserved in the `outputs/` directory for seamless project reporting.
