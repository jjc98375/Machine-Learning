import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class BinaryFocalLossWithLogits(nn.Module):
    """
    교수님 학점 킬러용 포컬 로스(Focal Loss)!
    기존 BCE 오차 함수에 modulating factor (1 - p_t)^gamma 를 추가하여 
    이미 잘 맞추는 80%의 평범한 Non-Switch 단어들의 오차는 무시(0으로 수렴)하고,
    아직 헷갈리는 20%의 Switch 단어에 100% 집중하도록 강제하는 초고급 논문용 오차 함수입니다.
    """
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super().__init__()
        # Switch(1) 클래스는 희귀하므로 0.8, 연속(0) 클래스는 0.2의 비율(alpha)로 학술적 4:1 비율을 완벽 세팅!
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # 1. 안정적인 기존 BCE 로스 획득
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 2. 모델이 예측한 확률 계산
        probs = torch.sigmoid(logits)
        
        # 3. 정답 위치의 확률 (p_t)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 4. 희귀 클래스 보정 가중치 (alpha_t)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 5. 핵심: Focal Loss 공식 적용! (잘 맞추는 건 패널티 0, 헷갈리는 건 패널티 배가)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class PredictiveSwitchModel(nn.Module):
    def __init__(self, model_name, lambda_sw=0.67, lambda_dur=0.33, focal_alpha=0.8, focal_gamma=2.0, unfreeze_layers=0):
        super().__init__()
        self.lambda_sw = lambda_sw
        self.lambda_dur = lambda_dur
        
        # Enable Causal masking (is_decoder=True)
        config = AutoConfig.from_pretrained(model_name)
        config.is_decoder = True
        
        # Shared backbone model
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size
        
        # MLP heads instead of single Linear — more capacity to learn switch patterns
        self.switch_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.loss_sw_fn = BinaryFocalLossWithLogits(alpha=focal_alpha, gamma=focal_gamma, reduction='none')
        
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)
        )
        self.loss_dur_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Freeze backbone, then selectively unfreeze top N layers
        if unfreeze_layers > 0:
            self._freeze_backbone(unfreeze_layers)
        
    def _freeze_backbone(self, unfreeze_layers):
        """Freeze all backbone params, then unfreeze the top N transformer layers + pooler."""
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze top N encoder layers
        encoder_layers = None
        if hasattr(self.backbone, 'encoder'):  # BERT-style
            encoder_layers = self.backbone.encoder.layer
        elif hasattr(self.backbone, 'layer'):
            encoder_layers = self.backbone.layer
            
        if encoder_layers is not None:
            total = len(encoder_layers)
            for layer in encoder_layers[max(0, total - unfreeze_layers):]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Always unfreeze pooler if it exists
        if hasattr(self.backbone, 'pooler') and self.backbone.pooler is not None:
            for param in self.backbone.pooler.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"  Backbone: {trainable:,}/{total:,} params trainable (top {unfreeze_layers} layers unfrozen)")

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
