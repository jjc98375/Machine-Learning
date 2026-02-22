"""
dataset.py — PyTorch IterableDataset that streams from HuggingFace,
             applies language ID, creates shifted labels, aligns to subwords,
             and yields model-ready batches.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, Optional

from config import PAIR_FILES, MODEL_NAME, MAX_LENGTH, MIN_SCORE
from data_loading import load_pair, extract_text
from labeling import identify_token_languages, create_shifted_labels, align_labels_to_subwords


class CompleteStreamingDataset(IterableDataset):
    """
    Streams SwitchLingua data and yields fully labeled samples.

    Each yielded sample contains:
      input_ids       (256,)  — subword token IDs
      attention_mask  (256,)  — 1=real token, 0=padding
      switch_labels   (256,)  — 0=no switch, 1=switch, -100=ignore
      duration_labels (256,)  — 0=Small, 1=Medium, 2=Large, -100=ignore
      lang_pair       str     — e.g. "Hindi-English"
    """

    def __init__(self, model_name=MODEL_NAME, max_length=MAX_LENGTH,
                 min_score=MIN_SCORE, pairs=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.min_score = min_score
        self.pairs = pairs or PAIR_FILES

    def _process_sample(self, text, lang1, lang2, pair_name) -> Optional[Dict]:
        """Full pipeline: text → language ID → labels → subword alignment."""
        words, langs = identify_token_languages(text, lang1, lang2)
        if len(words) < 3:
            return None

        y_sw, y_dur = create_shifted_labels(langs)
        aligned = align_labels_to_subwords(
            self.tokenizer, words, y_sw, y_dur, self.max_length
        )
        aligned["lang_pair"] = pair_name
        return aligned

    def _stream_pair(self, pair_name):
        """Stream and process samples from one CSV file."""
        lang1, lang2 = pair_name.split("-")
        ds = load_pair(pair_name, streaming=True)

        for sample in ds:
            score = sample.get("score")
            if score is not None and float(score) < self.min_score:
                continue
            text = extract_text(sample.get("data_generation_result", ""))
            if not text or len(text.split()) < 3:
                continue

            result = self._process_sample(text, lang1, lang2, pair_name)
            if result:
                yield result

    def __iter__(self):
        """Round-robin across all pairs for mixed batches."""
        streams = {p: iter(self._stream_pair(p)) for p in self.pairs}
        active = list(streams.keys())

        while active:
            exhausted = []
            for pair in active:
                try:
                    yield next(streams[pair])
                except StopIteration:
                    exhausted.append(pair)
            for p in exhausted:
                active.remove(p)


def collate_fn(batch: list) -> dict:
    """Stack tensors, keep strings as lists."""
    result = {}
    for k in ["input_ids", "attention_mask", "switch_labels", "duration_labels"]:
        if k in batch[0]:
            result[k] = torch.stack([s[k] for s in batch])
    for k in ["lang_pair"]:
        if k in batch[0]:
            result[k] = [s[k] for s in batch]
    return result


def build_dataloader(batch_size=32, **kwargs) -> DataLoader:
    """Build a ready-to-use streaming DataLoader."""
    ds = CompleteStreamingDataset(**kwargs)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn,
                      num_workers=0, pin_memory=torch.cuda.is_available())
