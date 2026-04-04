# Experiment Analysis: Groups A & C Results

**Purpose:** Guide Group D zero-shot universal model training. Any agent reading this should understand what worked, what didn't, and what config to use.

---

## Reliable Results (Groups A & C only)

Group B results are excluded from this analysis. Group B's alpha=0.5/gamma=0.0 run produced degenerate 1.0 F1 on same-script Latin pairs (French, German, Italian, Spanish), indicating the model collapsed to "always predict no-switch." Group B's alpha=0.8/gamma=2.0 and alpha=0.9/gamma=3.0 runs also showed partial degeneracy in mBERT. These results are unreliable for guiding next steps.

### Group A: Epoch & Sample Size Study

| Config | Model | Mean F1 | Sigma | Dur Acc | Switch F1 |
|--------|-------|---------|-------|---------|-----------|
| ep=3, s=2000, bs=32, lr=2e-5 | XLM-R | 0.668 | 0.096 | 0.641 | 0.558 |
| ep=3, s=2000, bs=32, lr=2e-5 | **mBERT** | **0.680** | **0.096** | **0.647** | **0.566** |
| ep=2, s=500, bs=32, lr=2e-5 | XLM-R | 0.609 | 0.099 | 0.608 | 0.496 |
| ep=2, s=500, bs=32, lr=2e-5 | mBERT | 0.638 | 0.092 | 0.625 | 0.524 |

**Findings:**
- More data (2000 vs 500) and more epochs (3 vs 2) consistently improve all metrics
- mBERT > XLM-R in both configurations
- Diminishing returns likely beyond 3000 samples/pair given the dataset size

### Group C: Batch Size & Learning Rate Study

| Config | Model | Mean F1 | Sigma | Dur Acc | Switch F1 |
|--------|-------|---------|-------|---------|-----------|
| ep=4, s=3000, bs=16, lr=1e-5 | XLM-R | 0.682 | 0.103 | 0.649 | 0.566 |
| ep=4, s=3000, bs=16, lr=1e-5 | mBERT | 0.687 | 0.100 | 0.651 | 0.571 |
| ep=4, s=3000, bs=64, lr=5e-5 | XLM-R | 0.686 | 0.103 | 0.655 | 0.577 |
| ep=4, s=3000, bs=64, lr=5e-5 | **mBERT** | **0.697** | **0.098** | **0.651** | **0.581** |

**Findings:**
- Larger batch (64) + higher LR (5e-5) slightly outperforms smaller batch (16) + lower LR (1e-5)
- The improvement is marginal (~0.01), model is not sensitive to these hyperparams
- mBERT wins in every configuration

---

## Consistent Patterns Across All Experiments

### 1. mBERT > XLM-R (always)
mBERT outperforms XLM-R in every single honest run. Margin is small (0.01-0.015 F1) but perfectly consistent. Use mBERT for Group D.

### 2. Per-Pair Performance Clusters
Pairs fall into two clear tiers based on script type:

**Tier 1 - High F1 (0.70-0.85): Different-script pairs**
| Pair | Avg F1 (mBERT, across A&C) | Script Difference |
|------|---------------------------|-------------------|
| Chinese-English | 0.837 | CJK vs Latin |
| Japanese-English | 0.822 | Kana/Kanji vs Latin |
| Korean-English | 0.741 | Hangul vs Latin |
| Hindi-English | 0.747 | Devanagari vs Latin |
| Russian-English | 0.703 | Cyrillic vs Latin |
| Arabic-English | 0.698 | Arabic vs Latin |

**Tier 2 - Low F1 (0.55-0.60): Same-script pairs**
| Pair | Avg F1 (mBERT, across A&C) | Script Difference |
|------|---------------------------|-------------------|
| German-English | 0.589 | Latin vs Latin |
| Italian-English | 0.585 | Latin vs Latin |
| French-English | 0.580 | Latin vs Latin |
| Spanish-English | 0.560 | Latin vs Latin |

**Interpretation:** When both languages share Latin script, the model must rely on vocabulary/morphology cues rather than script boundaries. This is fundamentally harder for subword tokenizers.

### 3. Universality (Sigma) is Stable
Sigma ranges from 0.092-0.103 across all honest runs. It's not sensitive to hyperparameters. The variance comes from the structural script-type gap, not model quality.

### 4. Duration Accuracy Plateaus at ~0.65
Duration accuracy is consistently 0.64-0.66 across all configs. More epochs/data don't help much. This may be a ceiling for the 3-class binning approach.

---

## Recommended Config for Group D (Zero-Shot Universal Training)

### Model
**mBERT** (`bert-base-multilingual-cased`)

### Hyperparameters
```
epochs: 3
samples_per_pair: 2000
batch_size: 32
learning_rate: 2e-5
focal_alpha: 0.8
focal_gamma: 2.0
lambda_sw: 0.67
lambda_dur: 0.33
```

**Rationale (Updated for Colab Usage Limits):**
- mBERT: consistent winner across all experiments
- 3 epochs & 2000 samples/pair: Group A proved this is the "sweet spot" for securing high performance (0.680 F1) while vastly reducing total runtime.
- bs=32, lr=2e-5: Completely prevents Google Colab T4 Out-Of-Memory (OOM) crashes and severely cuts down Compute Unit consumption. (Group C verified that pushing to bs=64/3000 samples only provides a marginal ~0.01 improvement, which is not worth the risk of crashing mid-training).

### Zero-Shot Setup
- **Train on 8 pairs:** Hindi-English, Arabic-English, Korean-English, Chinese-English, German-English, Italian-English, Russian-English, Japanese-English
- **Hold out 2 pairs for zero-shot eval:** French-English, Spanish-English
- **Why these hold-outs:** Both are same-script Latin pairs (the hardest category). If the model achieves non-trivial F1 on these unseen pairs, it proves genuine linguistic generalization, not script-boundary memorization.
- **Alternative hold-out option:** Hold out one different-script (e.g., Arabic-English) and one same-script (e.g., Spanish-English) for a more balanced test.

### What to Report for Group D
1. In-domain F1 (8 training pairs)
2. Zero-shot F1 (2 held-out pairs)
3. Compare zero-shot F1 to Group A/C's supervised F1 on the same pairs
4. Sigma computed across ALL 10 pairs (including zero-shot) to measure true universality

### Success Criteria
- Zero-shot F1 on held-out pairs > 0.50 (above random chance) = model generalizes
- Zero-shot F1 within 0.10 of supervised F1 on same pairs = strong universality claim
- Overall sigma < 0.12 across all 10 pairs = competitive with supervised baseline
