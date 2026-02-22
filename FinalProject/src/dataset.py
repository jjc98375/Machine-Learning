import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import pandas as pd
import numpy as np

class SwitchLinguaDataset(Dataset):
    def __init__(self, split="train", max_length=128, model_name="xlm-roberta-base"):
        """
        Args:
            split (str): 'train', 'validation', or 'test'
            max_length (int): Maximum sequence length
            model_name (str): Pretrained tokenizer name
        """
        self.dataset = load_dataset("Shelton1013/SwitchLingua_text", split=split)
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        tokens = item['tokens'] # List of words
        lid_tags = item['lid']  # List of language tags (e.g., 'en', 'es', 'mix')
        
        # 1. Label Engineering for PREDICTION (Shifted labels)
        # Input at t needs to predict if t+1 switches language.
        
        # Identify switch points in the ORIGINAL word sequence
        # We need to know: for each word, does the NEXT word start a new language block?
        
        # Just simple check: lid[i] != lid[i+1] ?
        # But we also need duration: how many tokens does the new language last?
        
        word_labels = []
        word_durations = []
        
        for i in range(len(tokens) - 1): # Can't predict for the very last token
            current_lang = lid_tags[i]
            next_lang = lid_tags[i+1]
            
            if current_lang != next_lang:
                # Switch occurs at i+1
                is_switch = 1
                
                # Calculate duration of the NEW language segment starting at i+1
                duration_count = 0
                for j in range(i+1, len(tokens)):
                    if lid_tags[j] == next_lang:
                        duration_count += 1
                    else:
                        break
                
                # Categorize duration
                if duration_count <= 2:
                    duration_class = 0 # Small
                elif duration_count <= 6:
                    duration_class = 1 # Medium
                else:
                    duration_class = 2 # Large
            else:
                is_switch = 0
                duration_class = -100 # Ignore index
                
            word_labels.append(is_switch)
            word_durations.append(duration_class)
            
        # Append dummy label for the last word (since we can't predict next)
        # Or just truncate the input. 
        # Strategy: Input is tokens[:-1], Label is switch_status of tokens[1:]
        
        # 2. Tokenization and Subword Alignment
        # This is tricky. We have word-level labels but subword tokens.
        # We generally assign the label to the FIRST subword of a word.
        
        encoding = self.tokenizer(
            tokens[:-1], # Inputs are all words except the last one (nothing to predict after it)
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Align labels to subwords
        word_ids = encoding.word_ids() # e.g. [None, 0, 0, 1, 2, 2, ...]
        
        subword_switch_labels = []
        subword_duration_labels = []
        
        for word_id in word_ids:
            if word_id is None:
                # Special tokens
                subword_switch_labels.append(-100)
                subword_duration_labels.append(-100)
            elif word_id >= len(word_labels):
                 # Truncation case or edge case
                subword_switch_labels.append(-100)
                subword_duration_labels.append(-100)
            else:
                # For a given word, is it the start of a prediction? 
                # Actually, we want to predict AT every token.
                # If a word is split into [sub1, sub2], both should probably carry the prediction
                # that the NEXT word will be a switch?
                # OR, only the last subword of the current word predicts the switch?
                # "I am go- -ing" -> "go" predicts switch? "ing" predicts switch?
                # Usually we align to first token, but for prediction, maybe valid for all?
                # Let's stick to: First subword carries label, others ignored (-100).
                
                # BUT wait, this is PREDICTION.
                # If I am at "go", I predict next.
                # If I am at "ing", I predict next.
                # The next word doesn't change. So technically all subwords could predict it.
                # However, to avoid double counting, let's just mask all but first subword for now.
                # To be consistent with standard NER/POS tasks.
                
                # We need to track if we've seen this word_id before in this sequence
                # Actually word_ids() allows easy check.
                # We'll rely on a 'seen' set or just check previous
                pass 
                
        # Re-loop to fill properly
        previous_word_id = None
        final_switch_labels = []
        final_duration_labels = []
        
        for word_id in word_ids:
            if word_id is None:
                final_switch_labels.append(-100) # -100 is PyTorch's ignore_index
                final_duration_labels.append(-100)
            elif word_id != previous_word_id:
                # First subword of this word
                if word_id < len(word_labels):
                    final_switch_labels.append(word_labels[word_id])
                    final_duration_labels.append(word_durations[word_id])
                else:
                    final_switch_labels.append(-100)
                    final_duration_labels.append(-100)
            else:
                # Subsequent subwords of the same word
                # We verify if we want to predict at every subword or just once per word.
                # Let's ignore subsequent subwords to not bias metric towards long words.
                final_switch_labels.append(-100)
                final_duration_labels.append(-100)
            
            previous_word_id = word_id

        # Convert to tensor
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels_switch': torch.tensor(final_switch_labels),
            'labels_duration': torch.tensor(final_duration_labels)
        }
        
        return item
