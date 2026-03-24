import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from baseline import compute_anticipatory_f1

def evaluate_model(model, dataloader, device):
    model.eval()
    all_true_sw = []
    all_pred_sw = []
    all_pairs = []
    
    all_true_dur = []
    all_pred_dur = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            switch_labels = batch["switch_labels"].to(device)
            duration_labels = batch["duration_labels"].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            # Switch predictions: sigmoid > 0.5
            switch_probs = torch.sigmoid(outputs["switch_logits"])
            switch_preds = (switch_probs > 0.5).long()
            
            # Duration predictions: argmax
            dur_preds = torch.argmax(outputs["duration_logits"], dim=-1)
            
            # Move to cpu for sklearn
            switch_labels = switch_labels.cpu().numpy()
            switch_preds = switch_preds.cpu().numpy()
            duration_labels = duration_labels.cpu().numpy()
            dur_preds = dur_preds.cpu().numpy()
            
            for i in range(len(batch["lang_pair"])):
                pair = batch["lang_pair"][i]
                
                # Active tokens
                active = switch_labels[i] != -100
                all_true_sw.extend(switch_labels[i][active])
                all_pred_sw.extend(switch_preds[i][active])
                all_pairs.extend([pair] * active.sum())
                
                # Duration evaluation
                active_dur = duration_labels[i] != -100
                all_true_dur.extend(duration_labels[i][active_dur])
                all_pred_dur.extend(dur_preds[i][active_dur])

    # 1. Switch Metrics (reuse phase1 function)
    switch_results = compute_anticipatory_f1(all_true_sw, all_pred_sw, all_pairs)
    
    # 2. Duration Metrics
    dur_acc = 0.0
    if len(all_true_dur) > 0:
        dur_acc = accuracy_score(all_true_dur, all_pred_dur)
        
    switch_results["duration_accuracy"] = dur_acc
    
    return switch_results

def print_evaluation_report(results, model_name):
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'='*60}")
    print(f"  Macro F1:         {results.get('f1_macro', 0):.4f}")
    print(f"  F1 (switch=1):    {results.get('f1_switch', 0):.4f}")
    print(f"  F1 (no_switch=0): {results.get('f1_no_switch', 0):.4f}")
    if "duration_accuracy" in results:
        print(f"  Duration Acc:     {results['duration_accuracy']:.4f}")
    print(f"\n  Per-Pair F1:")
    if "per_pair_f1" in results:
        for pair, f1 in results["per_pair_f1"].items():
            print(f"    {pair:<20s}: {f1:.4f}")
        print(f"\n  Universality σ: {results.get('sigma', 0):.4f}")
        print(f"  Mean F1:        {results.get('mean_f1', 0):.4f}")

def compare_models(results_dict):
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"| {'Model':<15} | {'Mean F1':<10} | {'Univ. σ':<10} | {'Dur. Acc':<10} |")
    print(f"|{'-'*17}|{'-'*12}|{'-'*12}|{'-'*12}|")
    for name, res in results_dict.items():
        mean_f1 = res.get('mean_f1', 0)
        sigma = res.get('sigma', 0)
        dur_acc = res.get('duration_accuracy', 0)
        print(f"| {name:<15} | {mean_f1:<10.4f} | {sigma:<10.4f} | {dur_acc:<10.4f} |")
