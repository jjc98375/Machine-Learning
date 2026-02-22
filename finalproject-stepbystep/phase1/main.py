"""
main.py — Run the entire Phase 1 pipeline.

This is the ONLY file you need to run:
  python main.py

It will:
  1. Explore all 6 language pairs (get stats for presentation)
  2. Demo the labeling pipeline on example sentences
  3. Verify the causal attention mask
  4. Test the complete streaming DataLoader
  5. Run the naive baseline and compute Anticipatory F1

File structure:
  config.py        — Constants and settings
  data_loading.py  — HuggingFace streaming + text extraction
  labeling.py      — Language ID + shifted labels + subword alignment
  causal_mask.py   — Causal attention mask
  dataset.py       — PyTorch streaming Dataset + DataLoader
  baseline.py      — Naive baseline + F1 computation
  main.py          — This file (runs everything)

Prerequisites:
  pip install datasets transformers torch tqdm langid scikit-learn
  huggingface-cli login
"""

import torch
from transformers import AutoTokenizer

from config import MODEL_NAME, MAX_LENGTH
from data_loading import explore_all_pairs
from labeling import identify_token_languages, create_shifted_labels, align_labels_to_subwords
from causal_mask import create_causal_mask, apply_causal_mask
from dataset import CompleteStreamingDataset, build_dataloader
from baseline import run_naive_baseline


def test_language_id():
    """TEST 1: Show token-level language identification on example sentences."""
    print("\n" + "=" * 60)
    print("[TEST 1] Token-Level Language Identification")
    print("=" * 60)

    examples = [
        ("I went to बाज़ार में to buy groceries", "Hindi", "English"),
        ("الطقس جميل today so let's go outside", "Arabic", "English"),
        ("오늘 weather가 정말 nice하네", "Korean", "English"),
        ("Let's go to the supermercado for groceries", "Spanish", "English"),
    ]

    for text, l1, l2 in examples:
        tokens, langs = identify_token_languages(text, l1, l2)
        print(f"\n  [{l1}-{l2}] {text}")
        for t, l in zip(tokens, langs):
            marker = "←" if l == l1 else ""
            print(f"    {t:<20s} → {l} {marker}")


def test_shifted_labels():
    """TEST 2: Show y_sw and y_dur creation step by step."""
    print("\n" + "=" * 60)
    print("[TEST 2] Shifted Label Creation (y_sw and y_dur)")
    print("=" * 60)

    text = "I went to बाज़ार में to buy groceries"
    tokens, langs = identify_token_languages(text, "Hindi", "English")
    y_sw, y_dur = create_shifted_labels(langs)

    sw_map = {0: "0(no)", 1: "1(YES!)", -100: "—(skip)"}
    dur_map = {0: "Small", 1: "Medium", 2: "Large", -100: "—"}

    print(f"\n  {'Pos':<5} {'Token':<15} {'Lang':<10} {'y_sw':<12} {'y_dur':<10}")
    print(f"  {'-'*52}")
    for i, (tok, lang, sw, dur) in enumerate(zip(tokens, langs, y_sw, y_dur)):
        print(f"  {i:<5} {tok:<15} {lang:<10} {sw_map[sw]:<12} {dur_map[dur]:<10}")

    print(f"\n  Summary:")
    print(f"    Total tokens:  {len(tokens)}")
    print(f"    Switch points: {sum(1 for s in y_sw if s == 1)}")
    print(f"    No-switch:     {sum(1 for s in y_sw if s == 0)}")
    print(f"    Masked (-100): {sum(1 for s in y_sw if s == -100)}")


def test_subword_alignment():
    """TEST 3: Show how word labels become subword labels."""
    print("\n" + "=" * 60)
    print("[TEST 3] Subword Label Alignment")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    text = "I went to बाज़ार में to buy groceries"
    tokens, langs = identify_token_languages(text, "Hindi", "English")
    y_sw, y_dur = create_shifted_labels(langs)

    # Use short max_length so output is readable
    aligned = align_labels_to_subwords(tokenizer, tokens, y_sw, y_dur, max_length=32)

    subtokens = tokenizer.convert_ids_to_tokens(aligned["input_ids"].tolist())
    sw_labels = aligned["switch_labels"].tolist()
    dur_labels = aligned["duration_labels"].tolist()

    print(f"\n  Words: {len(tokens)} → Subwords: {sum(1 for t in subtokens if t != '<pad>')}")
    print(f"\n  {'Pos':<5} {'Subword':<20} {'y_sw':<8} {'y_dur':<8} {'Note'}")
    print(f"  {'-'*55}")
    for i, (tok, sw, dur) in enumerate(zip(subtokens, sw_labels, dur_labels)):
        if tok == "<pad>":
            break
        if tok in ("<s>", "</s>"):
            note = "special token → ignored"
        elif sw >= 0:
            note = "★ SWITCH!" if sw == 1 else "first subword → real label"
        else:
            note = "continuation → masked"
        print(f"  {i:<5} {tok:<20} {sw:<8} {dur:<8} {note}")


def test_causal_mask():
    """TEST 4: Verify the causal attention mask works correctly."""
    print("\n" + "=" * 60)
    print("[TEST 4] Causal Attention Mask Verification")
    print("=" * 60)

    mask = create_causal_mask(6)
    print("\n  Causal mask (6x6):")
    for i, row in enumerate(mask.int().tolist()):
        sees = sum(row)
        print(f"    token {i}: {row}  (sees {int(sees)} tokens)")

    # Verify key properties
    assert mask[0].sum() == 1, "token 0 should only see itself"
    assert mask[5].sum() == 6, "token 5 should see all 6"
    assert mask[2][3] == 0, "token 2 should NOT see token 3 (future)"
    assert mask[3][2] == 1, "token 3 SHOULD see token 2 (past)"
    print("\n  ✅ All assertions passed!")

    # Test with padding
    attn = torch.tensor([[1, 1, 1, 1, 0, 0]])  # last 2 = padding
    combined = apply_causal_mask(create_causal_mask(6), attn)
    print(f"\n  With padding mask [1,1,1,1,0,0]:")
    print(f"    token 2 → token 3 (future):  {combined[0,0,2,3].item():.0f}  (blocked ✓)")
    print(f"    token 3 → token 2 (past):    {combined[0,0,3,2].item():.0f}  (allowed ✓)")
    print(f"    token 2 → token 5 (padding): {combined[0,0,2,5].item():.0f}  (blocked ✓)")
    print(f"    (0 = attend, -10000 = blocked)")


def test_complete_dataloader():
    """TEST 5: Verify the complete streaming DataLoader produces correct output."""
    print("\n" + "=" * 60)
    print("[TEST 5] Complete Streaming DataLoader")
    print("=" * 60)

    ds = CompleteStreamingDataset()
    sample = next(iter(ds))

    print(f"\n  First sample from stream:")
    print(f"    input_ids:       {sample['input_ids'].shape}")
    print(f"    attention_mask:  {sample['attention_mask'].shape}")
    print(f"    switch_labels:   {sample['switch_labels'].shape}")
    print(f"    duration_labels: {sample['duration_labels'].shape}")
    print(f"    lang_pair:       {sample['lang_pair']}")

    sw = sample["switch_labels"]
    print(f"\n  Label breakdown:")
    print(f"    -100 (masked):   {(sw == -100).sum().item()}")
    print(f"    0 (no switch):   {(sw == 0).sum().item()}")
    print(f"    1 (switch):      {(sw == 1).sum().item()}")


def test_naive_baseline():
    """TEST 6: Run naive baseline and compute Anticipatory F1."""
    print("\n" + "=" * 60)
    print("[TEST 6] Naive Baseline + Anticipatory F1")
    print("=" * 60)

    ds = CompleteStreamingDataset()
    # 3000 samples = ~500 per pair, takes ~5-10 min on MacBook Air
    # Reduce to 600 (~100 per pair) for a quick test
    results = run_naive_baseline(iter(ds), max_samples=3000)
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PHASE 1: COMPLETE PIPELINE")
    print("=" * 70)
    print("""
    File structure:
      config.py        — Constants (pairs, Unicode ranges, model settings)
      data_loading.py  — HuggingFace streaming + text extraction
      labeling.py      — Language ID + labels + subword alignment
      causal_mask.py   — Causal attention mask
      dataset.py       — PyTorch Dataset + DataLoader
      baseline.py      — Naive baseline + F1 metrics
      main.py          — This file (runs everything)
    """)

    # --- STEP 1: Explore dataset (uncomment to get stats for presentation) ---
    # explore_all_pairs(max_per_pair=2000)

    # --- STEP 2: Test each component ---
    test_language_id()
    test_shifted_labels()
    test_subword_alignment()
    test_causal_mask()
    test_complete_dataloader()

    # --- STEP 3: Run naive baseline (takes ~5-10 min) ---
    test_naive_baseline()

    # --- DONE ---
    print(f"\n{'='*70}")
    print("✅ PHASE 1 COMPLETE — Ready for Phase 2 (model training)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
