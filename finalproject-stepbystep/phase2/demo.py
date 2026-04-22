import os
import argparse
import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from phase2_config import MODELS
from model import PredictiveSwitchModel
from train import get_device


def _adapt_heads_to_checkpoint(model, state_dict):
    """
    Older checkpoints save switch_head / duration_head as plain nn.Linear
    (keys: `switch_head.weight`, `switch_head.bias`).
    Current model.py uses a 3-layer MLP (keys: `switch_head.0.weight`, ...).
    Swap the heads to plain Linear so the checkpoint loads cleanly.
    """
    is_simple_switch = (
        'switch_head.weight' in state_dict
        and state_dict['switch_head.weight'].ndim == 2
        and state_dict['switch_head.weight'].shape[0] == 1
    )
    is_simple_duration = (
        'duration_head.weight' in state_dict
        and state_dict['duration_head.weight'].ndim == 2
        and state_dict['duration_head.weight'].shape[0] == 3
    )
    if is_simple_switch and is_simple_duration:
        hidden = state_dict['switch_head.weight'].shape[1]
        model.switch_head = nn.Linear(hidden, 1)
        model.duration_head = nn.Linear(hidden, 3)
        print("ℹ️  Detected legacy single-Linear heads — adapted model architecture.")
    return model

def run_demo(model, tokenizer, device):
    print("\n" + "="*60)
    print("LIVE STREAMING CODE-SWITCHING PREDICTOR")
    print("="*60)
    print("Type a bilingual sentence word-by-word or paste a full sentence.")
    print("The model will predict the probability of a language switch occurring BEFORE the next word.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            text = input("\nEnter text: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if text.lower() in ['exit', 'quit']:
            break
            
        words = text.strip().split()
        if not words:
            continue
            
        print("\n[Simulation Started]")
        current_prefix = []
        
        for i, word in enumerate(words):
            current_prefix.append(word)
            prefix_text = " ".join(current_prefix)
            
            # Tokenize prefix
            inputs = tokenizer(prefix_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask)
                
                # Get the prediction for the VERY LAST token of the prefix
                # This predicts if the NEXT token will be a switch
                switch_logits = outputs["switch_logits"]
                last_token_logit = switch_logits[0, -1]
                prob = torch.sigmoid(last_token_logit).item()
                
                # Check what it predicts for duration
                dur_logits = outputs["duration_logits"][0, -1]
                dur_pred = torch.argmax(dur_logits).item()
                dur_text = ["Small (1-2 tokens)", "Medium (3-6 tokens)", "Large (7+ tokens)"][dur_pred]
                
            prob_percent = prob * 100
            alert = "🚨 HIGH PROBABILITY!" if prob > 0.5 else ""
            
            print(f"Step {i+1}: \"{prefix_text}\"")
            print(f"    ↳ Next-Token Switch Prob: {prob_percent:5.1f}% {alert}")
            if prob > 0.5:
                print(f"    ↳ Anticipated Duration:   {dur_text}")
            
        print("[Simulation Ended]\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, choices=["xlm-roberta", "mbert"], default="xlm-roberta")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pt file")
    args = parser.parse_args()
    
    device = torch.device('cpu')
    model_name = MODELS[args.backbone]
    
    print(f"Loading '{args.backbone}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PredictiveSwitchModel(model_name)

    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        model = _adapt_heads_to_checkpoint(model, state_dict)
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Error: Model path '{args.model_path}' not found!")
        exit(1)
        
    model.to(device)
    model.eval()
    
    run_demo(model, tokenizer, device)
