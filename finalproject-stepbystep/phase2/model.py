import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class PredictiveSwitchModel(nn.Module):
    def __init__(self, model_name, lambda_sw=1.0, lambda_dur=0.5):
        super().__init__()
        self.lambda_sw = lambda_sw
        self.lambda_dur = lambda_dur
        
        # Enable Causal masking (is_decoder=True)
        config = AutoConfig.from_pretrained(model_name)
        config.is_decoder = True
        
        # Shared backbone model
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size
        
        # Dual heads for multi-task
        self.switch_head = nn.Linear(hidden_size, 1)
        self.loss_sw_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        self.duration_head = nn.Linear(hidden_size, 3)
        self.loss_dur_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self, input_ids, attention_mask, switch_labels=None, duration_labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use last hidden state for token-level predictions
        hidden_states = outputs.last_hidden_state
        
        # Switch prediction
        switch_logits = self.switch_head(hidden_states).squeeze(-1) # (batch, seq_len)
        
        # Duration prediction
        duration_logits = self.duration_head(hidden_states) # (batch, seq_len, 3)
        
        loss = None
        loss_sw_val = 0.0
        loss_dur_val = 0.0
        
        if switch_labels is not None and duration_labels is not None:
            active_mask = (switch_labels != -100)
            if active_mask.sum() > 0:
                active_logits = switch_logits[active_mask]
                active_labels = switch_labels[active_mask].float()
                
                # BCE에 -100이 들어가면 간혹 연산 오버플로우로 NaN이 터지므로 필터 후 손실 계산
                batch_loss_sw = self.loss_sw_fn(active_logits, active_labels)
                loss_sw_val = batch_loss_sw.mean()
            else:
                loss_sw_val = torch.tensor(0.0, device=switch_logits.device, requires_grad=True)
                
            # Duration Loss (CE auto-ignores ignore_index=-100)
            loss_dur_val = self.loss_dur_fn(duration_logits.view(-1, 3), duration_labels.view(-1))
            if torch.isnan(loss_dur_val):
                loss_dur_val = torch.tensor(0.0, device=duration_logits.device, requires_grad=True)

            # Total Loss
            loss = self.lambda_sw * loss_sw_val + self.lambda_dur * loss_dur_val
            
        return {
            "loss": loss,
            "loss_sw": loss_sw_val.item() if isinstance(loss_sw_val, torch.Tensor) else loss_sw_val,
            "loss_dur": loss_dur_val.item() if isinstance(loss_dur_val, torch.Tensor) else loss_dur_val,
            "switch_logits": switch_logits,
            "duration_logits": duration_logits
        }
