# 🔮 Predictive Code-Switching Anticipation in Bilingual Discourse

## 🔥 About the Project
This repository hosts the final codebase for our culminating research project. We have engineered a **predictive** machine learning pipeline designed to *anticipate* **Code-Switching** boundaries (swaps in spoken language) and their consecutive durations within bilingual text.

Unlike traditional detection systems, our model operates in a strictly **causal streaming** environment. Given token $t$, the model must anticipate whether the *unseen* token $t+1$ initiates a foreign language swap.

Operating on a dual-head multitask architecture grafted onto **mBERT**, our network achieved profound universality across **10 typologically diverse language pairs** spanning Latin, Cyrillic, Devanagari, Arabic, and CJK scripts — while a dedicated data-saturation negative control (Group E) empirically established that the same-script Latin ceiling is a **representational** property of subword tokenization, not a training-budget artifact.

---

## 🚀 Key Technological Features
* **Multitask Shared Backbone:** Jointly predicts both binary language switching and multi-class switch durations to act as theoretical burstiness regularizers.
* **Causal Masking:** Standard bidirectional attention is aggressively masked via lower-triangular injection to prevent the model from cheating by reading the future.
* **Focal Loss Regularization:** Corrects extreme 4:1 class imbalances within conversational sequences, punishing the model for over-guessing "no-switch".
* **Zero-Shot Universality:** Flawless zero-shot predictive transfer on fully held-out Latin-script languages (F1 = 0.593 on French without a single French training token — within 0.003 of supervised ceiling).
* **Data-Saturation Negative Control:** Scaling to 30K samples/pair × 10 epochs does *not* cross the Latin ceiling; it *sharpens* it (Group E), proving the limitation is representational.

---

## 📊 Headline Results

| Paradigm | Config | Mean F1 | σ (universality) |
|---|---|---|---|
| Group A (hyperparameter scan) | ep3, s2000, mBERT | 0.680 | 0.096 |
| Group C (bs/lr sweep) | bs64, lr5e-5, mBERT | 0.697 | 0.098 |
| **Group D — Max Supervised** | **ep10, s10K, mBERT** | **0.707** | 0.100 |
| Group E — Scale Ceiling (XLM-R) | ep10, s10K, bs16 | 0.660 | 0.140 |
| Group E — Scale Ceiling (mBERT) | ep10, **s30K**, bs64 | 0.662 | 0.134 |
| **Zero-Shot (4 held-out)** | 6-pair train | **0.607** | **0.025** |

**Per-pair F1 (Group D, mBERT):** Chinese 0.869 · Japanese 0.845 · Korean 0.767 · Hindi 0.760 · Russian 0.721 · Arabic 0.719 · German 0.610 · Italian 0.606 · French 0.590 · Spanish 0.583.

### The Latin Ceiling & the Macro-vs-Mean F1 Divergence
Group E tripled Latin-script training exposure to 30K samples/pair; all four Latin pairs (De/It/Fr/Es) collapsed to **F1 = 0.5000 exactly** — the signature of a switch head defaulting to "always no-switch". This drove a diagnostic divergence between the two standard F1 aggregations:

* **Token-level Macro F1** (Switch / No-Switch averaged globally): **0.703 → 0.788 (+0.085)**
* **Mean Anticipatory F1** (per-pair macro averaged across 10 pairs): **0.707 → 0.662 (−0.045)**

Groups A, C, D kept both metrics within 0.02 of each other. Group E is the first configuration where they diverge by 0.126, and the divergence itself is evidence that Macro F1 *rewards* majority-class collapse on difficult pairs — the paper therefore uses **Mean F1** and **σ_universality** as the primary cross-lingual metrics.

---

## 🛠 Project Structure

```
finalproject-stepbystep/
├── main.tex                      ACM-format research paper (5 experimental groups)
├── PRESENTATION_DRAFT.md         Defense script
├── phase1/                       Data pipeline
├── phase2/                       Model, training, evaluation, demo
├── final/outputs/                Experimental deliverables (JSON per run)
│   ├── Group A/                  Epoch/sample-size ablation
│   ├── Group B/                  Focal loss α/γ sweep (not in paper)
│   ├── Group C/                  Batch-size / learning-rate sweep
│   ├── GroupD/                   Max supervised + zero-shot + single-task
│   └── GroupE/                   Scale-ceiling negative control (NEW)
├── slide_group_e_scale_ceiling.html     Presentation slide — D vs E heatmap
├── slide_macro_mean_divergence.html     Presentation slide — Macro/Mean gap
└── README.md
```

### `phase1/` — Data Pipeline & Baseline Architecture
* **`dataset.py`**: Low-memory streaming wrapper loading the 420K `SwitchLingua` dataset from the HuggingFace Hub, with local disk caching for fast re-runs.
* **`labeling.py`**: Binary switch alignment and duration stratification with Unicode-range verification (CJK/Arabic) and deterministic `langid` classification.

### `phase2/` — Model Training & CLI Execution
* **`model.py`**: Neural core. Loads `bert-base-multilingual-cased`, applies the causal mask (`is_decoder=True`), and bridges to two 3-layer MLP classification heads. Supports `single_task=True` ablation.
* **`train.py`**: AdamW optimization loop with cosine-annealing LR, early-stopping patience, periodic `_ckpt.pt` saves, and label smoothing on the duration head.
* **`run_experiment.py`**: Master CLI for hyperparameter-decoupled experiments.
* **`evaluate.py`** / **`aggregate_results.py`**: Evaluation harness and JSON summary writer.
* **`qualitative_analysis.py`**: Token-level qualitative predictions (used for paper's error-analysis section).
* **`demo.py`**: Interactive streaming predictor (see below).
* **`visualize.py`**: Per-pair F1 heatmap generation.

---

## 💻 How to Run Training

Navigate into `phase2/` and use the unified CLI:

```bash
# Quick validation (2 epochs, 500 samples)
python run_experiment.py --epochs 2 --samples_per_pair 500

# Group D: max-supervision ceiling (mBERT, full fine-tune)
python run_experiment.py --epochs 10 --samples_per_pair 10000 \
  --batch_size 32 --lr 2e-5 --unfreeze_layers 0

# Group E: scale-ceiling negative control (mBERT, 30K/pair)
python run_experiment.py --epochs 10 --samples_per_pair 30000 \
  --batch_size 64 --lr 2e-5 --unfreeze_layers 0

# Zero-shot generalization (holds out 4 pairs)
python run_experiment.py --epochs 5 --samples_per_pair 5000 \
  --zero_shot_pairs French-English Spanish-English Chinese-English Japanese-English

# Resume from a checkpoint (HPC crash recovery)
python run_experiment.py --resume_path outputs/my_previous_run/_ckpt.pt

# Single-task Switch-only ablation (duration loss disabled)
python run_experiment.py --single_task
```

All models, result JSONs, and visualization plots are archived under `phase2/outputs/<run_id>/` with hyperparameter-stamped naming to prevent overwriting.

---

## 🎬 How to Run the Interactive Demo

The demo streams a bilingual sentence word-by-word and prints the anticipated next-token switch probability.

```bash
cd phase2

# mBERT checkpoint (Group C-style run, trained on 10 pairs)
python3 demo.py --backbone mbert \
  --model_path ../final/mbert_run_ep4_s3000_bs64_lr5e-05_a0.8_g2.0_final.pt

# Then type a bilingual sentence and press Enter, e.g.:
#   We should probably just aller au magasin
#   我觉得这个 decision is really important
# Type 'exit' or Ctrl+D to quit.
```

**Expected behavior** (consistent with paper's qualitative analysis):
* **CJK → English switches (e.g. 我觉得这个 → decision):** Strong signal (~65–70 %).
* **English → Romance switches (aller / pero / mañana):** Moderate signal (~55–60 %), reflecting the Latin-script ceiling.
* **Loanword false positives (e.g. jalapeños):** Mild spike just below threshold.

The demo auto-detects legacy single-`nn.Linear` heads and the current 3-layer MLP heads, so checkpoints from either architecture generation load cleanly. Checkpoint `.pt` files are not committed — train locally or download from the project artifacts.

---

## 📂 Experimental Outputs (`final/outputs/`)

Every F1 value in the paper is reproducible from these JSONs. Schema per file:

```json
{
  "run_id": "...",
  "hyperparameters": { "epochs": ..., "samples": ..., "batch_size": ..., ... },
  "in_domain": {
    "<backbone>": {
      "f1_macro": ...,       # token-level Macro F1 (Switch + No-Switch averaged)
      "f1_switch": ...,      # global F1 for the minority class
      "f1_no_switch": ...,   # global F1 for the majority class
      "duration_accuracy": ...,
      "per_pair_f1": { "Chinese-English": ..., ... },
      "sigma": ...,          # universality σ across pairs
      "mean_f1": ...         # per-pair averaged Anticipatory F1
    }
  },
  "zero_shot": { ... }       # populated only for zero-shot runs
}
```

Groups A/B/C/D cover the paper's ablations and maximum-supervision run. **Group E** adds the two data-saturation runs (10 epochs × 10K XLM-R, 10 epochs × 30K mBERT) that bracket the representational ceiling from above.

---

## 📝 Document Archive
* **`main.tex`** — Completed ACM-format empirical research paper (5 experimental groups, zero-shot analysis, Macro-vs-Mean F1 divergence, full limitations & future-work).
* **`PRESENTATION_DRAFT.md`** — Scripting structure for the final oral defense.
* **`slide_group_e_scale_ceiling.html`** — D-vs-E heatmap slide showing Latin collapse.
* **`slide_macro_mean_divergence.html`** — Macro/Mean F1 divergence visualization.

---

## 📚 Citation & Dataset
Trained on **SwitchLingua** (Xie et al., NeurIPS 2025 Datasets & Benchmarks) — the first large-scale multilingual and multi-ethnic code-switching dataset. Backbone models: `bert-base-multilingual-cased` and `xlm-roberta-base` via HuggingFace.
