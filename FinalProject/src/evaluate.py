import torch
from torch.utils.data import DataLoader
from .dataset import SwitchLinguaDataset
from .model import PredictiveSwitchModel
from sklearn.metrics import f1_score
import numpy as np
import argparse

def evaluate(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load Model
    model = PredictiveSwitchModel()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load Test Data
    # Ideally we load the test split
    dataset = SwitchLinguaDataset(split='test', max_length=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    print("Running Inference...")
    
    all_preds_switch = []
    all_labels_switch = []
    
    # Note: To calculate universality per language pair, we need to know the language pair of each sample.
    # The current dataset loader might not expose this easily in __getitem__.
    # We might need to modify dataset logic to return 'lang_pair' metadata.
    # For now, let's assume we can access it from the dataset object if we iterate sequentially 
    # OR we just calculate global F1.
    # TO FIX: We need language pair info.
    
    # Let's iterate index-wise to get metadata
    
    results_by_lang_pair = {} # { 'eng-spa': {'preds': [], 'labels': []} }
    
    with torch.no_grad():
        for i in range(len(dataset)):
            # Get raw item for metadata
            raw_item = dataset.dataset[i]
            # Assumed metadata field for language pair. 
            # Looking at SwitchLingua on HF, usually has a configuration or field.
            # If not present, we might have to infer from tokens/lid.
            # Let's assume a 'language_pair' column exists or similar.
            # If not, we might group by the set of languages in 'lid'.
            
            # Fallback: Just use 'unknown' if not found, to at least run.
            lang_pair = raw_item.get('language_pair', 'unknown') 
            
            # Prepare tensor
            processed_item = dataset[i]
            input_ids = processed_item['input_ids'].unsqueeze(0).to(device)
            attention_mask = processed_item['attention_mask'].unsqueeze(0).to(device)
            labels_switch = processed_item['labels_switch'].unsqueeze(0)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs['switch_logits'] # [1, SeqLen]
            preds = (torch.sigmoid(logits) > 0.5).long().cpu()
            
            # Mask active
            active = (labels_switch != -100)
            valid_preds = preds[active]
            valid_labels = labels_switch[active]
            
            if lang_pair not in results_by_lang_pair:
                results_by_lang_pair[lang_pair] = {'preds': [], 'labels': []}
            
            results_by_lang_pair[lang_pair]['preds'].extend(valid_preds.tolist())
            results_by_lang_pair[lang_pair]['labels'].extend(valid_labels.tolist())
            
    # Calculate Universality
    f1_scores = []
    print("\n--- Results per Language Pair ---")
    for lp, data in results_by_lang_pair.items():
        if len(data['labels']) > 0:
            f1 = f1_score(data['labels'], data['preds'], average='binary') # Focus on Switch=1 class
            f1_scores.append(f1)
            print(f"{lp}: F1 = {f1:.4f}")
    
    if f1_scores:
        avg_f1 = np.mean(f1_scores)
        std_dev = np.std(f1_scores)
        print("\n--- Universality Metric ---")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Standard Deviation (sigma): {std_dev:.4f}")
        print("(Lower sigma is better)")
    else:
        print("No valid predictions found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1) # keeping simple for loop
    args = parser.parse_args()
    evaluate(args)
