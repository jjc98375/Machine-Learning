"""
labeling.py — The core label engineering pipeline.
  Step 1: identify_token_languages()  → figure out each word's language
  Step 2: create_shifted_labels()     → create y_sw and y_dur
  Step 3: align_labels_to_subwords()  → map word labels to subword tokens
"""

from collections import Counter
from typing import Dict, List, Tuple
import torch
from config import SCRIPT_RANGES, LANG_TO_SCRIPT, LANG_TO_ISO


# =============================================================================
# STEP 1: TOKEN-LEVEL LANGUAGE IDENTIFICATION
# =============================================================================

def detect_script(word: str) -> str:
    """Detect the Unicode script of a word (e.g. "बाज़ार" → "Devanagari")."""
    counts = Counter()
    for ch in word:
        cp = ord(ch)
        for name, ranges in SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] += 1
                break
    return counts.most_common(1)[0][0] if counts else "Unknown"


def identify_token_languages(
    text: str, lang1: str, lang2: str
) -> Tuple[List[str], List[str]]:
    """
    Assign a language to every word in the text.

    For different-script pairs (Hindi-EN, Arabic-EN, Korean-EN, Chinese-EN):
      → Uses Unicode script detection. Fast and accurate.

    For same-script pairs (Spanish-EN, French-EN):
      → Both use Latin alphabet, so uses langid library as fallback.

    Args:
        text:  "I went to बाज़ार में to buy groceries"
        lang1: "Hindi"
        lang2: "English"

    Returns:
        tokens: ["I", "went", "to", "बाज़ार", "में", "to", "buy", "groceries"]
        langs:  ["English", "English", "English", "Hindi", "Hindi", ...]
    """
    tokens = text.strip().split()
    if not tokens:
        return [], []

    s1 = LANG_TO_SCRIPT.get(lang1, "Unknown")
    s2 = LANG_TO_SCRIPT.get(lang2, "Unknown")

    # --- DIFFERENT SCRIPTS: use Unicode detection ---
    if s1 != s2:
        langs, prev = [], lang2
        for tok in tokens:
            clean = tok.strip(".,!?;:\"'()[]{}—–-·…""''")
            if not clean or clean.isdigit():
                langs.append(prev)
                continue
            script = detect_script(clean)
            if script == s1:
                lang = lang1
            elif script == s2:
                lang = lang2
            else:
                lang = prev
            langs.append(lang)
            prev = lang
        return tokens, langs

    # --- SAME SCRIPT: use langid library ---
    try:
        import langid
        langid.set_languages([LANG_TO_ISO[lang1], LANG_TO_ISO[lang2]])
        langs, prev = [], lang2
        for tok in tokens:
            clean = tok.strip(".,!?;:\"'()[]{}—–-·…""''")
            if not clean or len(clean) < 3 or clean.isdigit():
                langs.append(prev)
                continue
            iso, _ = langid.classify(clean)
            lang = lang1 if iso == LANG_TO_ISO[lang1] else lang2
            langs.append(lang)
            prev = lang
        return tokens, langs
    except ImportError:
        if not identify_token_languages._langid_warned:
            identify_token_languages._langid_warned = True
            print("WARNING: pip install langid (needed for same-script pairs)")
        return tokens, [lang2] * len(tokens)

# Module-level flag for one-time warning
identify_token_languages._langid_warned = False


# =============================================================================
# STEP 2: SHIFTED LABEL CREATION
# =============================================================================

def create_shifted_labels(
    token_langs: List[str],
) -> Tuple[List[int], List[int]]:
    """
    Create prediction labels from language tags.

    y_sw[t]:  Will token t+1 be a different language?
              0 = no switch, 1 = switch, -100 = can't predict (last token)

    y_dur[t]: If there's a switch at t+1, how long will it last?
              0 = Small (1-2 words), 1 = Medium (3-6), 2 = Large (7+)
              -100 = no switch here (question doesn't apply)
    """
    n = len(token_langs)

    # Pre-compute segment lengths (how many consecutive words share a language)
    seg_lens = {}
    i = 0
    while i < n:
        start, lang = i, token_langs[i]
        while i < n and token_langs[i] == lang:
            i += 1
        seg_lens[start] = i - start

    y_sw, y_dur = [], []
    for t in range(n):
        if t == n - 1:
            # Last token: no next word to predict
            y_sw.append(-100)
            y_dur.append(-100)
        elif token_langs[t] != token_langs[t + 1]:
            # Switch! Language changes at t+1
            y_sw.append(1)
            seg = seg_lens.get(t + 1, 1)
            y_dur.append(0 if seg <= 2 else 1 if seg <= 6 else 2)
        else:
            # No switch. Same language continues.
            y_sw.append(0)
            y_dur.append(-100)  # duration only defined at switch points

    return y_sw, y_dur


# =============================================================================
# STEP 3: SUBWORD LABEL ALIGNMENT
# =============================================================================

def align_labels_to_subwords(
    tokenizer,
    words: List[str],
    y_sw: List[int],
    y_dur: List[int],
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Align word-level labels to XLM-RoBERTa subword tokens.

    Rule: first subword of each word → real label
          continuation subwords     → -100 (ignored in loss)
          special tokens / padding  → -100

    Returns dict with: input_ids, attention_mask, switch_labels, duration_labels
    """
    enc = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    wids = enc.word_ids(batch_index=0)
    sw_aligned, dur_aligned = [], []
    prev = None

    for wid in wids:
        if wid is None:
            sw_aligned.append(-100)
            dur_aligned.append(-100)
        elif wid != prev:
            sw_aligned.append(y_sw[wid] if wid < len(y_sw) else -100)
            dur_aligned.append(y_dur[wid] if wid < len(y_dur) else -100)
        else:
            sw_aligned.append(-100)
            dur_aligned.append(-100)
        prev = wid

    return {
        "input_ids":       enc["input_ids"].squeeze(0),
        "attention_mask":  enc["attention_mask"].squeeze(0),
        "switch_labels":   torch.tensor(sw_aligned, dtype=torch.long),
        "duration_labels": torch.tensor(dur_aligned, dtype=torch.long),
    }
