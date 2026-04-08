# Agent Task: Final Project Deliverables for "The Universal Predictor"

## Context

You are helping complete the final deliverables for an NEU Machine Learning course final project. The project is **"Predictive Multitask Learning for Streaming Code-Switching: The Universal Predictor"** — a system that predicts when bilingual speakers will switch languages mid-conversation and how long the switch will last.

**Deadline:** April 22, 2026 (In-Class Final Defense)
**Format:** 15-minute Final Defense (12 min presentation + 3 min Q&A)

---

## Project Summary

- **Task:** Given a bilingual text prefix (x1,...,xt), predict (1) whether token t+1 will switch languages (binary), and (2) the duration of the switch segment (Small/Medium/Large).
- **Architecture:** Dual-head model on top of multilingual transformer backbones (mBERT and XLM-RoBERTa). Switch Head uses Focal Loss (alpha=0.8, gamma=2.0); Duration Head uses CrossEntropy.
- **Dataset:** SwitchLingua_text (420K bilingual samples), 10 language pairs spanning Latin-script, CJK, Cyrillic, Arabic, Devanagari, and Hangul.
- **Key Innovation:** Zero-shot universality — train on N pairs, predict on unseen pairs.

---

## Experiment Pipeline & Backbone Selection

We ran experiments in a structured pipeline: Groups A-C were exploratory/ablation studies comparing both backbones (mBERT and XLM-RoBERTa), which informed the backbone choice for the final Group D runs.

### Groups A & C: Backbone Comparison (mBERT vs XLM-RoBERTa)

**Group A — Epoch & Sample Size Study:**

| Config | Model | Mean F1 | Sigma | Dur Acc | Switch F1 |
|--------|-------|---------|-------|---------|-----------|
| ep=3, s=2000, bs=32, lr=2e-5 | XLM-R | 0.668 | 0.096 | 0.641 | 0.558 |
| ep=3, s=2000, bs=32, lr=2e-5 | **mBERT** | **0.680** | **0.096** | **0.647** | **0.566** |
| ep=2, s=500, bs=32, lr=2e-5 | XLM-R | 0.609 | 0.099 | 0.608 | 0.496 |
| ep=2, s=500, bs=32, lr=2e-5 | mBERT | 0.638 | 0.092 | 0.625 | 0.524 |

**Group C — Batch Size & Learning Rate Study:**

| Config | Model | Mean F1 | Sigma | Dur Acc | Switch F1 |
|--------|-------|---------|-------|---------|-----------|
| ep=4, s=3000, bs=16, lr=1e-5 | XLM-R | 0.682 | 0.103 | 0.649 | 0.566 |
| ep=4, s=3000, bs=16, lr=1e-5 | mBERT | 0.687 | 0.100 | 0.651 | 0.571 |
| ep=4, s=3000, bs=64, lr=5e-5 | XLM-R | 0.686 | 0.103 | 0.655 | 0.577 |
| ep=4, s=3000, bs=64, lr=5e-5 | **mBERT** | **0.697** | **0.098** | **0.651** | **0.581** |

**Backbone Decision:** mBERT outperformed XLM-R in **every single configuration** across Groups A and C (margin ~0.01-0.015 F1). This was consistent across all metrics (Mean F1, Switch F1, Duration Accuracy) and all hyperparameter settings. Based on this evidence, **mBERT was selected as the sole backbone for Group D's final zero-shot and max-training runs.**

(Note: Group B focal loss ablation results were excluded due to degenerate outputs — model collapsed to "always predict no-switch" under certain alpha/gamma settings.)

### Group D: Final Runs (mBERT only)

**The Ultimate Supervised Run:** mBERT, 10 epochs, 10K samples/pair, all 10 pairs
- **Mean F1:** 0.707 across all 10 pairs
- **CJK ceiling:** ~0.84-0.86 F1
- **Latin ceiling:** ~0.58-0.61 F1
- **Duration Accuracy:** plateaus ~0.65

**Zero-Shot Run:** mBERT, 5 epochs, 5K samples, trained on 6 pairs, held out 4 (French, Spanish, Chinese, Japanese)
- Latin pairs (French/Spanish) Zero-Shot F1: 0.593 / 0.572 — effectively matching supervised ceiling
- CJK pairs (Chinese/Japanese) Zero-Shot F1: 0.640 / 0.621 — high for unseen distant scripts
- Zero-Shot Universality Sigma: 0.025

### Key Findings
- **mBERT > XLM-R always** — proven across every Group A & C config, which justified using mBERT exclusively for Group D
- Different-script pairs (Chinese, Japanese, Korean, Hindi) consistently outperform same-script pairs (French, Spanish, German, Italian)
- Focal Loss (alpha=0.8, gamma=2.0) addresses the 4:1 class imbalance
- Hyperparameter sensitivity is low — batch size and LR have marginal effects

---

## Codebase Structure (for Technical Implementation criterion)

- **`dataset.py`**: Streaming HuggingFace loader with subword-to-label alignment and causal masking. Handles gated SwitchLingua dataset via hub streaming to avoid local storage limits.
- **`model.py`**: Dual-head architecture — Switch Head (Focal Loss, BCE) and Duration Head (CrossEntropy) on transformer backbone. Dynamic token filters prevent -100 padding overflow.
- **`train.py`**: AdamW optimizer with linear warmup, per-epoch checkpoint saving, automatic hardware detection (CUDA/MPS/CPU).
- **`run_experiment.py`**: Centralized CLI controller for batch experiments. Accepts `--epochs`, `--samples_per_pair`, `--batch_size`, `--lr`, `--focal_alpha`, `--focal_gamma`, `--backbones`, `--zero_shot_pairs`. Outputs go to timestamped run folders.
- **`visualize.py`**: Matplotlib loss convergence curves and per-pair F1 bar charts.
- **`baseline.py`**: Naive prediction baselines and evaluation math (Anticipatory F1, Macro F1).
- **Libraries:** PyTorch, HuggingFace Transformers, HuggingFace Datasets (streaming mode), Matplotlib
- **Reproducibility:** Every run auto-saves to a named folder (e.g., `run_ep4_s3000_bs32_lr2e-05_a0.8_g2.0/`) containing model weights, plots, performance JSONs, and `experiment_history.txt`.

---

## Grading Rubric (30 points total)

### 1. Progress from Project Update 2 (5 pts)
- **Exceptional (5):** Clearly addresses Update 2 feedback. Significant advancements in all aspects.
- **Good (4.5):** Addresses most feedback. Noticeable advancements.
- **Average (2.5):** Some progress but feedback not fully addressed.
- **Poor (1.5):** Little to no meaningful progress.

### 2. Methodology & Approach (6 pts)
- **Exceptional (6):** Well-reasoned, appropriate methodology. Clearly justified methods. Strong understanding.
- **Good (5.4):** Described and generally justified. May lack some detail.
- **Average (3):** Weak/unclear justification. Basic understanding.
- **Poor (1.8):** Poorly described, lacks justification, inappropriate.

### 3. Results & Analysis (6 pts)
- **Exceptional (6):** Clear, well-organized results. Thorough, insightful analysis. Meaningful conclusions supported by evidence. Discusses limitations.
- **Good (5.4):** Generally clear results. Some analysis depth missing.
- **Average (3):** Unclear/disorganized results. Superficial analysis.
- **Poor (1.8):** Unclear, incomplete, or missing results.

### 4. Technical Implementation / Code (4 pts)
- **Exceptional (4):** Well-documented, structured, efficient, reproducible code. Proper use of tools/libraries.
- **Good (3.6):** Generally documented and structured. Minor improvements possible.
- **Average (2):** Poorly documented, inefficient. Inconsistent tool use.
- **Poor (1.2):** Poorly documented, unstructured, not reproducible.

### 5. Report Quality & Documentation (4 pts)
- **Exceptional (4):** Well-written, logically organized, comprehensive. Proper citations. Professional.
- **Good (3.6):** Well-organized, covers key aspects. Generally clear.
- **Average (2):** Disorganized, lacking detail. Unclear writing.
- **Poor (1.2):** Poorly written, incomplete, significant errors.

### 6. Presentation & Communication (5 pts)
- **Exceptional (5):** Clear, engaging, well-organized. Effective communication of technical details. Thoughtful Q&A answers. Clear visual aids.
- **Good (4.5):** Clear, logical. Adequate technical communication.
- **Average (2.5):** Disorganized, hard to follow. Unclear technical details.
- **Poor (1.5):** Fails to deliver clear presentation.

---

## Final Defense Checklist (10 points within presentation)

### Final Performance Synthesis (3 pts)
- Present final Anticipatory F1 and Duration Accuracy
- Explain whether the multitask approach improved understanding of switch "burstiness"
- Compare supervised vs zero-shot paradigms

### Universality Metric (sigma_universality) (3 pts)
- Report Standard Deviation of performance across all pairs
- Identify which model paradigm achieved the lowest variance
- Supervised sigma ~0.098 vs Zero-Shot sigma 0.025

### Qualitative Analysis (2 pts)
- Contrast "Successful Predictions" (True Positives) with "False Alarms" (False Positives)
- Analyze whether the model struggles more with inter-sentential or intra-sentential switches
- Provide concrete token-level examples

### Project Demo (1 pt)
- Brief real-time demo: model predicts likelihood and length of next switch from a bilingual prefix
- Show word-by-word streaming prediction

### Deliverables Confirmation (1 pt)
- Research Paper submitted
- GitHub Repo submitted
- Performance JSONs submitted

---

## What I Need You To Do

Using the rubric and results above, help me produce the following deliverables that maximize my score across all 6 rubric categories:

1. **Research Paper (PDF/LaTeX):** A complete final report covering:
   - Abstract, Introduction, Related Work, Methodology, Experiments (including ablation studies), Results & Analysis, Limitations, Conclusion
   - Must reference progress from Update 2 (expanded from 6 to 10 pairs, zero-shot evaluation, addressed professor feedback on loss weights and universality)
   - Include tables for all experiment groups (A, C, D) with F1, sigma, duration accuracy
   - Discuss the script-type gap (different-script vs same-script pairs) as a key finding
   - Proper academic citations

2. **Presentation Slides (content for 12 slides):** Following the defense checklist structure:
   - Title, Problem Statement, Progress Since Update 2, Data, Architecture, Training Details, Supervised Results, Zero-Shot Results, Universality Analysis, Qualitative Error Analysis, Demo Slide, Conclusion
   - Each slide should have speaker notes

3. **Performance JSON:** A structured JSON summarizing all final metrics for submission.

### Key Points to Emphasize for Maximum Score
- **Progress:** We went from "code complete, no results" (Update 2) to 7+ full experiment runs with ablation studies and zero-shot evaluation
- **Methodology:** Justify Focal Loss over BCE, dual-head multitask design, causal masking for anticipatory prediction, zero-shot leave-out protocol
- **Results:** The zero-shot universality sigma of 0.025 is the headline result — model generalizes across unseen language pairs
- **Limitations:** Same-script pairs plateau at ~0.58-0.61 F1 (subword tokenizer limitation), duration accuracy ceiling at ~0.65 (3-class binning may be too coarse)
- **Update 2 Feedback Addressed:** "more language pairs" (6->10), "one universal model" (zero-shot proves this), "loss weights sum to 1" (lambda_sw=0.67 + lambda_dur=0.33)
