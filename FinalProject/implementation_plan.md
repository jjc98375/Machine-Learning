# Predictive Code-Switching Detection Implementation Plan

## Goal Description
Build a real-time language switch prediction model that anticipates code-switching at the next token.
The model will perform two tasks simultaneously (Multitask Learning):
1.  **Switch Prediction**: Binary classification (Is next token a switch?).
2.  **Duration Prediction**: 3-class classification (Small: 1-2, Medium: 3-6, Large: 7+ tokens).

**Key Constraint**: Causal masking (cannot see future tokens).
**Success Metric**: Universality (minimizing standard deviation of F1-scores across 12+ language pairs).

## User Review Required
> [!IMPORTANT]
> **Model Selection**: The plan assumes using `XLM-RoBERTa` as the primary backbone as suggested in instructions, comparing against `mBERT`.
> **Framework**: I plan to use **PyTorch** and **Hugging Face Transformers**.
> **Compute**: Training transformers requires GPU. If local compute is limited, we might need to use Colab/Kaggle or smaller models for local debugging.

## Proposed Changes

### Structure
I propose organizing the code into a collaborative structure:
```
FinalProject/
├── src/
│   ├── dataset.py       # Data loading, tokenization, shifted label generation
│   ├── model.py         # Custom Transformer with 2 heads
│   ├── train.py         # Training loop using PyTorch/Lightning
│   └── evaluate.py      # Universality metric calculation
├── tests/
│   └── test_data.py     # Unit tests for label shifting logic
└── requirements.txt
```

### [src] Data Processing
#### [NEW] [dataset.py](file:///Users/joshcho/Documents/after%2015/NEU/Machine%20Learning/FinalProject/src/dataset.py)
- Load `Shelton1013/SwitchLingua_text`.
- **Label Shifting Logic**:
    - Input: Token sequence $t_0, t_1, ..., t_n$
    - Target for $t_i$: Whether $t_{i+1}$ represents a language switch.
    - Must handle subword tokenization carefully (align labels to first subword of next token).
- Implement `SwitchLinguaDataset` class.

### [src] Model Architecture
#### [NEW] [model.py](file:///Users/joshcho/Documents/after%2015/NEU/Machine%20Learning/FinalProject/src/model.py)
- Class `PredictiveSwitchModel(nn.Module)`:
    - Backbone: `xlm-roberta-base`.
    - Head 1 (Switch): Linear -> Sigmoid (Binary Cross Entropy).
    - Head 2 (Duration): Linear -> Softmax (Cross Entropy, ignored if no switch).
    - **Causal Masking**: Ensure the attention mask prevents attending to future tokens.

### [src] Training & Evaluation
#### [NEW] [train.py](file:///Users/joshcho/Documents/after%2015/NEU/Machine%20Learning/FinalProject/src/train.py)
- Standard PyTorch training loop.
- Loss function: $Loss = L_{switch} + \lambda L_{duration}$.
- Logging: Track "Universality" during validation if possible, or just avg F1.

#### [NEW] [evaluate.py](file:///Users/joshcho/Documents/after%2015/NEU/Machine%20Learning/FinalProject/src/evaluate.py)
- Load trained model.
- Run inference on test set.
- **Metric**:
    1. Group results by Language Pair.
    2. Calculate F1 for each pair.
    3. Compute Std Dev ($\sigma$) of F1s.

## Verification Plan

### Automated Tests
- **Label Shifting Unit Test**:
    - Create a small synthetic list of tokens and language tags.
    - Verify that the label for position $t$ matches the language change status of $t+1$.
    - Verify duration calculation.
    - Command: `pytest tests/test_data.py`

### Manual Verification
- **Overfitting Test**:
    - Train on a tiny subset (e.g., 100 samples).
    - specific command: `python src/train.py --debug --batch_size 2`
    - Verify training loss goes to near zero.
- **Universality Check**:
    - Run `evaluate.py` and manually check if the output CSV/Printout shows F1 per language pair.
