import argparse
import os
import torch
from collections import defaultdict
from train import get_device, ListDataset
from dataset import CompleteStreamingDataset, collate_fn
from model import PredictiveSwitchModel
from phase2_config import MODELS, PAIR_FILES, OUTPUT_DIR

def extract_examples(model, dataloader, device, tokenizer, max_examples=10):
    model.eval()
    
    examples = {
        "True_Positive": [], # Model predicted 1, Actual was 1
        "False_Positive": [], # Model predicted 1, Actual was 0 (False alarm)
        "False_Negative": [], # Model predicted 0, Actual was 1 (Missed switch)
    }
    
    print("\n🔍 정성적 분석(Qualitative Analysis)을 위해 문장을 스캔하는 중입니다...")
    
    with torch.no_grad():
        for batch in dataloader:
            if len(examples["True_Positive"]) >= max_examples and len(examples["False_Positive"]) >= max_examples:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            switch_labels = batch["switch_labels"].to(device)
            
            outputs = model(input_ids, attention_mask)
            switch_probs = torch.sigmoid(outputs["switch_logits"])
            switch_preds = (switch_probs > 0.5).long()
            
            switch_labels = switch_labels.cpu().numpy()
            switch_preds = switch_preds.cpu().numpy()
            input_ids = input_ids.cpu().numpy()
            
            for i in range(len(batch["lang_pair"])):
                pair = batch["lang_pair"][i]
                seq_len = sum(attention_mask[i].cpu().numpy())
                
                # Iterate through tokens in the sequence
                for t in range(1, seq_len - 1): # Ignore [CLS] and [SEP]
                    true_idx = switch_labels[i][t]
                    if true_idx == -100:
                        continue
                        
                    pred_idx = switch_preds[i][t]
                    
                    # Token sequence up to this point (prefix)
                    prefix_ids = input_ids[i][1:t+1]
                    next_id = input_ids[i][t+1]
                    
                    prefix_text = tokenizer.decode(prefix_ids)
                    next_text = tokenizer.decode([next_id])
                    
                    example_str = f"[{pair}] {prefix_text} ➡️ '{next_text}'\n"
                    example_str += f"   Model Probability: {switch_probs[i][t]:.4f}\n"
                    
                    category = None
                    if true_idx == 1 and pred_idx == 1:
                        category = "True_Positive"
                    elif true_idx == 0 and pred_idx == 1:
                        category = "False_Positive"
                    elif true_idx == 1 and pred_idx == 0:
                        category = "False_Negative"
                        
                    if category and len(examples[category]) < max_examples:
                        examples[category].append(example_str)

    return examples

def run_analysis(args):
    device = get_device()
    model_name = MODELS[args.backbone]
    
    print(f"Loading tokenizer & configuration for {args.backbone}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model weights...")
    model = PredictiveSwitchModel(model_name)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        print("✅ Weights loaded successfully!")
    else:
        print("⚠️ Warning: Model path not found! Using randomly initialized model for demo.")
        
    model.to(device)
    
    # 500개 샘플 수집하여 분석
    print("Collecting validation samples...")
    from train import get_eval_dataloader
    include_pairs = list(PAIR_FILES.keys())[:3] # 분석을 위해 상위 3개 언어쌍만 수집
    dataloader = get_eval_dataloader(model_name, 200, include_pairs)
    
    results = extract_examples(model, dataloader, device, tokenizer, max_examples=15)
    
    out_path = os.path.join(OUTPUT_DIR, f"{args.backbone}_qualitative_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"=== QUALITATIVE ANALYSIS REPORT ({args.backbone}) ===\n")
        
        for category, ex_list in results.items():
            f.write(f"\n[{category.upper()}] ({len(ex_list)} examples)\n")
            f.write("-" * 50 + "\n")
            for ex in ex_list:
                f.write(ex + "\n")
                
    print(f"\n✅ 정성적 분석 레포트 덤프 완료: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, choices=["xlm-roberta", "mbert"], default="xlm-roberta")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pt file")
    args = parser.parse_args()
    run_analysis(args)
