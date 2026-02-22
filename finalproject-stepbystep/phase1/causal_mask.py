"""
causal_mask.py — Causal attention mask to prevent future information leakage.
                 Used during model training (Phase 2), but defined and verified here.

Standard XLM-RoBERTa is bidirectional (token t sees ALL tokens).
We add a causal mask so token t can ONLY see tokens 0, 1, ..., t.
This prevents the model from "cheating" by peeking at future tokens.
"""

import torch


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create a lower-triangular causal attention mask.

    Example (seq_len=5):
      [[1, 0, 0, 0, 0],   ← token 0 sees only itself
       [1, 1, 0, 0, 0],   ← token 1 sees tokens 0-1
       [1, 1, 1, 0, 0],   ← token 2 sees tokens 0-2
       [1, 1, 1, 1, 0],   ← token 3 sees tokens 0-3
       [1, 1, 1, 1, 1]]   ← token 4 sees all past tokens
    """
    return torch.tril(torch.ones(seq_len, seq_len))


def apply_causal_mask(
    causal_mask: torch.Tensor,     # (seq_len, seq_len)
    attention_mask: torch.Tensor,  # (batch_size, seq_len)
) -> torch.Tensor:
    """
    Combine causal mask with padding mask for HuggingFace models.

    Token i can attend to token j ONLY IF:
      1. j <= i          (causal: no future tokens)
      2. j is not padding (padding mask)

    Returns: (batch_size, 1, seq_len, seq_len) tensor
      0.0 = attend,  -10000.0 = blocked
    """
    causal_4d = causal_mask.unsqueeze(0).unsqueeze(0)      # (1, 1, S, S)
    padding_4d = attention_mask.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, S)
    combined = causal_4d * padding_4d                       # (B, 1, S, S)
    return (1.0 - combined) * -10000.0
