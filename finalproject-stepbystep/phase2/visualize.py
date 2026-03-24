import matplotlib.pyplot as plt
import os
import numpy as np

def plot_f1_bar_chart(results_dict, save_path):
    models = list(results_dict.keys())
    
    pairs = []
    for m in models:
        for p in results_dict[m].get("per_pair_f1", {}).keys():
            if p not in pairs:
                pairs.append(p)
                
    if not pairs:
        print("No per-pair F1 data to plot.")
        return
        
    pairs.sort()
    
    x = np.arange(len(pairs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model in enumerate(models):
        f1s = [results_dict[model].get("per_pair_f1", {}).get(p, 0) for p in pairs]
        offset = width * i - (width * (len(models)-1) / 2)
        ax.bar(x + offset, f1s, width, label=model)
        
    ax.set_ylabel('Anticipatory F1 Score')
    ax.set_title('Per-Pair Anticipatory F1 by Model Backbone')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    ax.legend(loc='upper left')
    
    # Add sigma text to top right
    textstr = '\n'.join([f"{m} σ: {results_dict[m].get('sigma', 0):.4f}" for m in models])
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
            
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved bar chart to {save_path}")

def plot_convergence(history, model_name, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Switch Loss
    ax1.plot(epochs, history["train_loss_sw"], 'b-', label='Train')
    ax1.plot(epochs, history["val_loss_sw"], 'r-', label='Val')
    ax1.set_title(f'{model_name}\nSwitch Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Duration Loss
    ax2.plot(epochs, history["train_loss_dur"], 'b-', label='Train')
    ax2.plot(epochs, history["val_loss_dur"], 'r-', label='Val')
    ax2.set_title(f'{model_name}\nDuration Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved convergence plot to {save_path}")

def plot_comparison_convergence(h_xlmr, h_mbert, save_path):
    plt.figure(figsize=(8, 6))
    
    epochs_xlmr = range(1, len(h_xlmr["val_loss"]) + 1)
    epochs_mbert = range(1, len(h_mbert["val_loss"]) + 1)
    
    plt.plot(epochs_xlmr, h_xlmr["val_loss"], 'b-', label='XLM-R Val Loss')
    plt.plot(epochs_mbert, h_mbert["val_loss"], 'r-', label='mBERT Val Loss')
    
    plt.title('Validation Loss Comparison (Total)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison convergence plot to {save_path}")
