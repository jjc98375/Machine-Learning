import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
import os

from phase2_config import LR, EPOCHS, WARMUP_RATIO, BATCH_SIZE, MODELS_DIR, PAIR_FILES, MODELS
from dataset import CompleteStreamingDataset, collate_fn
from model import PredictiveSwitchModel

def collect_dataset(model_name, max_per_pair):
    print(f"Collecting up to {max_per_pair} samples per pair for {model_name}...")
    dataset_iter = CompleteStreamingDataset(model_name=model_name)
    pair_counts = defaultdict(int)
    collected_samples = []
    
    target_pairs = list(PAIR_FILES.keys())
    
    for sample in dataset_iter:
        pair = sample["lang_pair"]
        if pair_counts[pair] < max_per_pair:
            collected_samples.append(sample)
            pair_counts[pair] += 1
            
            if len(collected_samples) % 500 == 0:
                print(f"  Collected {len(collected_samples)} samples...")
                
        if all(pair_counts[p] >= max_per_pair for p in target_pairs):
            break
            
    print(f"Successfully collected {len(collected_samples)} total samples.")
    return collected_samples

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        return self.data_list[idx]

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_model(model_name, epochs=EPOCHS, max_samples_per_pair=2000, resume_path=None):
    device = get_device()
    print(f"Using device: {device}")
    
    # Get short model identifier
    model_id = [k for k, v in MODELS.items() if v == model_name][0]
    
    # Collect data
    samples = collect_dataset(model_name, max_samples_per_pair)
    dataset = ListDataset(samples)
    
    # Random split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = PredictiveSwitchModel(model_name)
    
    if resume_path and os.path.exists(resume_path):
        print(f"🔥 머리 여는 중... 저장된 과거의 뇌({resume_path})를 모델에 덮어씌웁니다!!")
        model.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))
        print("✅ 뇌 이식 완료! 남은 바퀴 수만큼 훈련을 추가로 시작합니다.")
        
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    
    history = {
        "train_loss": [], "train_loss_sw": [], "train_loss_dur": [],
        "val_loss": [], "val_loss_sw": [], "val_loss_dur": []
    }
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss, total_sw, total_dur = 0, 0, 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            switch_labels = batch["switch_labels"].to(device)
            duration_labels = batch["duration_labels"].to(device)
            
            outputs = model(input_ids, attention_mask, switch_labels, duration_labels)
            loss = outputs["loss"]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_sw += outputs["loss_sw"]
            total_dur += outputs["loss_dur"]
            
        avg_train_loss = total_loss / len(train_loader)
        avg_train_sw = total_sw / len(train_loader)
        avg_train_dur = total_dur / len(train_loader)
        
        # Validation
        model.eval()
        val_loss, val_sw, val_dur = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                switch_labels = batch["switch_labels"].to(device)
                duration_labels = batch["duration_labels"].to(device)
                
                outputs = model(input_ids, attention_mask, switch_labels, duration_labels)
                val_loss += outputs["loss"].item()
                val_sw += outputs["loss_sw"]
                val_dur += outputs["loss_dur"]
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_sw = val_sw / len(val_loader)
        avg_val_dur = val_dur / len(val_loader)
        
        history["train_loss"].append(avg_train_loss)
        history["train_loss_sw"].append(avg_train_sw)
        history["train_loss_dur"].append(avg_train_dur)
        history["val_loss"].append(avg_val_loss)
        history["val_loss_sw"].append(avg_val_sw)
        history["val_loss_dur"].append(avg_val_dur)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} (SW: {avg_train_sw:.4f}, DUR: {avg_train_dur:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} (SW: {avg_val_sw:.4f}, DUR: {avg_val_dur:.4f})")
              
    # Save model
    model_path = os.path.join(MODELS_DIR, f"{model_id}_final.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    return {
        "model": model,
        "history": history,
        "val_loader": val_loader
    }
