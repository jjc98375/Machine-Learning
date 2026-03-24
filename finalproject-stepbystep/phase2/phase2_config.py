import os
import sys
import importlib.util
from pathlib import Path

# Load phase1 config module explicitly to avoid circular import with current config.py
phase1_config_path = Path(__file__).parent.parent / "phase1" / "config.py"
spec = importlib.util.spec_from_file_location("phase1_config", phase1_config_path)
phase1_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase1_config)

PAIR_FILES = phase1_config.PAIR_FILES
MAX_LENGTH = phase1_config.MAX_LENGTH
BATCH_SIZE = phase1_config.BATCH_SIZE

# Add phase1 to sys.path so we can import dataset/baseline etc.
PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "phase1"))
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)


# Training Hyperparameters
LR = 2e-5
EPOCHS = 5
WARMUP_RATIO = 0.1

# Loss Weights
LAMBDA_SW = 1.0
LAMBDA_DUR = 0.5

# Backbones
MODELS = {
    "xlm-roberta": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased"
}

# Data Settings
MAX_SAMPLES_PER_PAIR = 2000

# Output Paths
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

for d in [OUTPUT_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)
