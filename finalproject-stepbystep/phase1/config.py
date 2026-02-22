"""
config.py — All constants, language pair settings, and Unicode tables.
             Import this from other files. Nothing runs here.
"""

# =============================================================================
# OUR 6 SELECTED LANGUAGE PAIRS
# =============================================================================
# Maps friendly name → CSV filename on HuggingFace
PAIR_FILES = {
    "Hindi-English":   "Hindi_eng.csv",     # Axis 1: Different Script
    "Arabic-English":  "Arabic_eng.csv",     # Axis 1: Different Script
    "Spanish-English": "Spanish_eng.csv",    # Axis 2: Same Script
    "French-English":  "French_eng.csv",     # Axis 2: Same Script
    "Korean-English":  "Korean_eng.csv",     # Axis 3: Distant Typology
    "Chinese-English": "Chinese_eng.csv",    # Axis 3: Distant Typology
}

# =============================================================================
# UNICODE SCRIPT RANGES (for language identification)
# =============================================================================
SCRIPT_RANGES = {
    "Arabic":     [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "CJK":        [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "Hangul":     [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
    "Latin":      [(0x0041, 0x024F)],
}

# Maps language name → script it uses
LANG_TO_SCRIPT = {
    "Hindi": "Devanagari", "Arabic": "Arabic", "Chinese": "CJK",
    "Korean": "Hangul", "Spanish": "Latin", "French": "Latin",
    "English": "Latin",
}

# Maps language name → ISO code (for langid library)
LANG_TO_ISO = {
    "Hindi": "hi", "Arabic": "ar", "Chinese": "zh", "Korean": "ko",
    "Spanish": "es", "French": "fr", "English": "en",
}

# =============================================================================
# MODEL SETTINGS
# =============================================================================
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 256
MIN_SCORE = 7.0
BATCH_SIZE = 32
