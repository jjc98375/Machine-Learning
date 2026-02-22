"""
baseline.py — Naive baseline evaluation and Anticipatory F1 computation.
              The naive baseline predicts "no switch" (0) for every token.
              This establishes the floor that our real model must beat.
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List
from sklearn.metrics import f1_score


def compute_anticipatory_f1(
    all_true: List[int],
    all_pred: List[int],
    all_pairs: List[str],
) -> Dict:
    """
    Compute Anticipatory F1 — how well we predict FUTURE switches.

    Returns overall F1, per-class F1, per-pair F1, and universality σ.
    """
    valid = [(t, p, pair) for t, p, pair in zip(all_true, all_pred, all_pairs) if t >= 0]
    if not valid:
        return {"error": "No valid predictions"}

    true = [v[0] for v in valid]
    pred = [v[1] for v in valid]

    results = {
        "f1_macro": f1_score(true, pred, average="macro", zero_division=0),
        "f1_switch": f1_score(true, pred, average="binary", pos_label=1, zero_division=0),
        "f1_no_switch": f1_score(true, pred, average="binary", pos_label=0, zero_division=0),
    }

    # Per-pair F1 for universality
    pair_data = defaultdict(lambda: {"t": [], "p": []})
    for t, p, pair in valid:
        pair_data[pair]["t"].append(t)
        pair_data[pair]["p"].append(p)

    pair_f1 = {pair: f1_score(d["t"], d["p"], average="macro", zero_division=0)
               for pair, d in sorted(pair_data.items())}

    results["per_pair_f1"] = pair_f1
    vals = list(pair_f1.values())
    results["sigma"] = float(np.std(vals))
    results["mean_f1"] = float(np.mean(vals))
    return results


def run_naive_baseline(dataset_iter, max_samples: int = 3000) -> Dict:
    """
    Run "always predict no switch" baseline on streamed data.
    Collects labels, computes F1, prints a full report.
    """
    all_true, all_pred, all_pairs = [], [], []
    sw_counts, dur_counts, pair_counts = Counter(), Counter(), Counter()
    n = 0

    print("Collecting samples from stream...")
    for sample in dataset_iter:
        for label in sample["switch_labels"].tolist():
            if label >= 0:
                all_true.append(label)
                all_pred.append(0)
                all_pairs.append(sample["lang_pair"])
                sw_counts[label] += 1

        for label in sample["duration_labels"].tolist():
            if label >= 0:
                dur_counts[label] += 1

        pair_counts[sample["lang_pair"]] += 1
        n += 1
        if n % 500 == 0:
            print(f"  {n:,} samples processed...")
        if n >= max_samples:
            break

    # --- Print report ---
    total = sum(sw_counts.values())
    print(f"\n{'='*60}")
    print(f"LABEL DISTRIBUTION ({n:,} samples, {total:,} valid tokens)")
    print(f"{'='*60}")
    for label in sorted(sw_counts):
        name = "no_switch" if label == 0 else "switch"
        print(f"  y_sw={label} ({name}): {sw_counts[label]:,} ({sw_counts[label]/total*100:.1f}%)")
    print(f"  Imbalance: 1:{sw_counts[0] // max(sw_counts.get(1, 1), 1)}")

    dur_names = {0: "Small(1-2)", 1: "Medium(3-6)", 2: "Large(7+)"}
    if dur_counts:
        td = sum(dur_counts.values())
        print(f"\n  Duration distribution:")
        for d in sorted(dur_counts):
            print(f"    {dur_names.get(d, d)}: {dur_counts[d]:,} ({dur_counts[d]/td*100:.1f}%)")

    print(f"\n  Samples per pair:")
    for pair, c in pair_counts.most_common():
        print(f"    {pair}: {c:,}")

    # --- Compute F1 ---
    print(f"\n{'='*60}")
    print("NAIVE BASELINE: 'Always Predict No Switch'")
    print(f"{'='*60}")
    results = compute_anticipatory_f1(all_true, all_pred, all_pairs)

    print(f"  Macro F1:         {results['f1_macro']:.4f}")
    print(f"  F1 (switch=1):    {results['f1_switch']:.4f}")
    print(f"  F1 (no_switch=0): {results['f1_no_switch']:.4f}")
    print(f"\n  Per-Pair F1:")
    for pair, f1 in results["per_pair_f1"].items():
        print(f"    {pair:<20s}: {f1:.4f}")
    print(f"\n  Universality σ: {results['sigma']:.4f}")
    print(f"  Mean F1:        {results['mean_f1']:.4f}")

    return results
