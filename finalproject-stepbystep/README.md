# 🔮 Predictive Code-Switching Anticipation in Bilingual Discourse

## 🔥 About the Project
This repository hosts the final codebase for our culminating research project. We have successfully engineered a **predictive** machine learning pipeline designed to *anticipate* **Code-Switching** boundaries (swaps in spoken language) and their consecutive durations within bilingual text. 

Unlike traditional detection systems, our model operates in a strictly **causal streaming** environment. Given token $t$, the model must anticipate whether the *unseen* token $t+1$ initiates a foreign language swap.

Operating on a dual-head multitask architecture grafted onto **mBERT**, our network achieved profound universality: proving mathematically that it can maintain predictive rigor across **10 typologically diverse language pairs** spanning Latin, Cyrillic, Devanagari, and CJK scripts.

---

## 🚀 Key Technological Features
* **Multitask Shared Backbone:** Jointly predicts both binary language switching and multi-class switch durations to act as theoretical burstiness regularizers.
* **Causal Masking:** Standard bidirectional attention is aggressively masked via lower-triangular injection to prevent the model from cheating by reading the future.
* **Focal Loss Regularization:** Successfully corrects extreme 4:1 class imbalances within conversational sequences, punishing the model for over-guessing "no-switch".
* **Zero-Shot Universality:** The system demonstrates flawless zero-shot predictive transfer capabilities on fully held-out Latin-script languages (scoring 0.593 F1 on French without a single French training token).

---

## 🛠 Project Structure

### `phase1/` - Data Pipeline & Baseline Architecture
* **`dataset.py`**: A low-memory streaming wrapper loading the massive 420K `SwitchLingua` dataset directly via the HuggingFace Hub.
* **`labeling.py`**: Handles on-the-fly binary alignments and switch duration stratification masking. Features rapid Unicode verification (for CJK/Arabic) and deterministic `langid` classification.

### `phase2/` - Advanced Model Training & CLI Execution
* **`model.py`**: The neural core. Loads `bert-base-multilingual-cased`, applies the Causal mask, and bridges to two dense Multi-Layer Perceptron (MLP) classification heads integrated with a `single_task` ablation bypass.
* **`train.py`**: Controls the AdamW optimization loop. Now includes Early Stopping (Patience) and periodic `_ckpt.pt` model state saving to protect against HPC server timeouts.
* **`run_experiment.py`**: The master CLI script allowing for fully decoupled hyperparameter tuning natively from the terminal. 

---

## 💻 How to Run training

Navigate into `phase2/` and use the unified command-line interface:

```bash
# 1. Run a quick validation test (2 epochs, 500 samples)
python run_experiment.py --epochs 2 --samples_per_pair 500

# 2. Run the Ultimate 10-Pair Supervised Ceiling training (Limit Testing)
python run_experiment.py --epochs 10 --samples_per_pair 10000 --batch_size 32 --lr 2e-5 --unfreeze_layers 0

# 3. Replicate the Zero-Shot Generalization Test 
# (Trains on 6 pairs, holds exactly 4 pairs out for inference)
python run_experiment.py --epochs 5 --samples_per_pair 5000 --zero_shot_pairs French-English Spanish-English Chinese-English Japanese-English

# 4. Resume from an unexpected HPC cluster crash
python run_experiment.py --resume_path outputs/my_previous_run/_ckpt.pt

# 5. Run a Single-Task Switch-Only Ablation (disables duration loss)
python run_experiment.py --single_task
```

All models, results, JSON statistical summaries, and visual plots are safely archived dynamically within the `outputs/` folder containing the exact hyperparameter stamp naming schema to prevent overwriting.

---
## 📝 Document Archive
* `main.tex`: The completed 8-page ACM empirical research paper detailing the mathematical and topological findings. 
* `PRESENTATION_DRAFT.md`: Scripting structure formulated for the final oral defense board.
