import argparse
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, get_scheduler
from tqdm import tqdm
from .dataset import SwitchLinguaDataset
from .model import PredictiveSwitchModel
import os

def train(args):
    # Setup Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load Data
    full_dataset = SwitchLinguaDataset(split='train', max_length=128)
    
    # Split Train/Val (Simple split for now)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Model
    model = PredictiveSwitchModel().to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    # Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_switch = batch['labels_switch'].to(device)
            labels_duration = batch['labels_duration'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask, labels_switch, labels_duration)
            loss = outputs['loss']
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_switch = batch['labels_switch'].to(device)
                labels_duration = batch['labels_duration'].to(device)
                
                outputs = model(input_ids, attention_mask, labels_switch, labels_duration)
                val_loss += outputs['loss'].item()
        
        print(f"Epoch {epoch+1} - Val Loss: {val_loss / len(val_loader):.4f}")
        
        # Save Checkpoint
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_ep{epoch+1}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    train(args)
