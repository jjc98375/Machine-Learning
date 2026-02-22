"""
data_loading.py — Load SwitchLingua CSV files and explore the dataset.
                  Handles HuggingFace connection, streaming, and text extraction.
"""

import json
import statistics
from collections import Counter
from typing import Optional
from datasets import load_dataset
from config import PAIR_FILES


def load_pair(pair_name: str, streaming: bool = True):
    """Load one language pair's CSV file from HuggingFace."""
    return load_dataset(
        "Shelton1013/SwitchLingua_text",
        data_files=PAIR_FILES[pair_name],
        streaming=streaming,
        split="train",
    )


def extract_text(raw) -> Optional[str]:
    """
    Extract text from the data_generation_result column.
    Handles 3 formats: plain string, JSON list string, or Python list.
    """
    if isinstance(raw, str):
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return " [SEP] ".join(str(t) for t in parsed)
            except json.JSONDecodeError:
                pass
        return raw
    if isinstance(raw, list):
        return " [SEP] ".join(str(t) for t in raw)
    return str(raw)


def explore_all_pairs(max_per_pair: int = 2000):
    """
    Stream each CSV file and collect stats for the presentation.
    Prints a summary table with samples, scores, text lengths per pair.
    """
    print("=" * 70)
    print("EXPLORING OUR 6 SELECTED LANGUAGE PAIRS")
    print("=" * 70)

    all_stats = {}

    for pair_name, filename in PAIR_FILES.items():
        print(f"\n--- Loading {pair_name} ({filename}) ---")
        try:
            ds = load_pair(pair_name, streaming=True)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        scores, text_lengths = [], []
        cs_types, conv_types = Counter(), Counter()
        total = 0

        for sample in ds:
            total += 1
            if total <= max_per_pair:
                s = sample.get("score")
                if s is not None:
                    scores.append(float(s))
                text = extract_text(sample.get("data_generation_result", ""))
                if text:
                    text_lengths.append(len(text.split()))
                cs_types[sample.get("cs_type", "unknown")] += 1
                conv_types[sample.get("conversation_type", "unknown")] += 1

        all_stats[pair_name] = {
            "total": total,
            "mean_score": statistics.mean(scores) if scores else 0,
            "mean_words": statistics.mean(text_lengths) if text_lengths else 0,
        }
        print(f"  Samples: {total:,} | Score: {all_stats[pair_name]['mean_score']:.2f} | Words: {all_stats[pair_name]['mean_words']:.1f}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Pair':<20} {'Samples':>8} {'Avg Score':>10} {'Avg Words':>10}")
    print("-" * 50)
    total_all = 0
    for pair, s in all_stats.items():
        print(f"{pair:<20} {s['total']:>8,} {s['mean_score']:>10.2f} {s['mean_words']:>10.1f}")
        total_all += s["total"]
    print("-" * 50)
    print(f"{'TOTAL':<20} {total_all:>8,}")
    return all_stats
