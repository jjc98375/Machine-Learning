import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig

class PredictiveSwitchModel(nn.Module):
    def __init__(self, model_name='xlm-roberta-base'):
        super().__init__()
        
        # Load Pretrained model
        # Use is_decoder=True to enforce causal masking (prevent peeking at future tokens)
        self.config = XLMRobertaConfig.from_pretrained(model_name)
        self.config.is_decoder = True 
        self.roberta = XLMRobertaModel.from_pretrained(model_name, config=self.config)
        
        hidden_size = self.config.hidden_size
        
        # Task 1: Switch Prediction (Binary)
        # 0 = No Switch, 1 = Switch
        self.switch_head = nn.Linear(hidden_size, 1)
        
        # Task 2: Duration Prediction (Multi-class)
        # 0 = Small, 1 = Medium, 2 = Large
        self.duration_head = nn.Linear(hidden_size, 3)
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none') # For switch
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')   # For duration

    def forward(self, input_ids, attention_mask, labels_switch=None, labels_duration=None):
        # Forward pass through Transformer
        # We don't need to pass decoder_input_ids because we are using it as a causal encoder 
        # on the input_ids themselves. 
        # Note: XLMRobertaModel acting as decoder usually expects exact shift if used in Enc-Dec,
        # but here we just want self-attention to be causal.
        
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state # [Batch, SeqLen, Hidden]
        
        # Heads
        switch_logits = self.switch_head(sequence_output).squeeze(-1) # [Batch, SeqLen]
        duration_logits = self.duration_head(sequence_output)         # [Batch, SeqLen, 3]
        
        loss = None
        if labels_switch is not None and labels_duration is not None:
            # Masking active parts (ignore padding and special tokens logic handled by -100 in labels)
            active_mask = (labels_switch != -100)
            
            # 1. Switch Loss
            # specific active elements
            active_switch_logits = switch_logits[active_mask]
            active_switch_labels = labels_switch[active_mask].float()
            
            loss_switch = self.bce_loss(active_switch_logits, active_switch_labels)
            loss_switch = loss_switch.mean()
            
            # 2. Duration Loss
            # Only calculated where switch == 1 AND valid label
            # We can just rely on ignore_index=-100 if we set standard CE, 
            # but our duration labels are -100 for non-switches too (from dataset logic).
            # Let's verify dataset logic:
            # If switch=0, duration=-100.
            # So standard CrossEntropy with ignore_index=-100 works perfectly.
            
            # Flatten for CE
            # duration_logits: [N, 3], labels: [N]
            loss_duration = self.ce_loss(
                duration_logits.view(-1, 3), 
                labels_duration.view(-1)
            )
            # This automatically ignores -100
            
            # Total Loss
            # Basic sum, or weighted?
            loss = loss_switch + loss_duration
            
        return {
            'loss': loss,
            'switch_logits': switch_logits,
            'duration_logits': duration_logits
        }
