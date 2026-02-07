"""
=========================================================
 Multimodal Fusion Model (Task 4 & 5) — v2 SIMPLIFIED
 
 Uses EXACT same LSTM(32) architecture as Task 1/2 (which
 achieved 76.72% binary) for the visual branch.
 Adds a simple rPPG MLP branch for early fusion.
 Supports transfer learning from Task 1/2 pretrained weights.
 
 Architecture:
   Visual:  (T, 9) → LSTM(32, 2L) → Attention → 32-dim → FC(32)
   rPPG:    (14,)  → FC(16) → 16-dim  
   Fused:   48-dim → FC(32) → heads (binary + 4-class + regression)
=========================================================
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Simple attention pooling (same as Task 1/2)."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        weighted_sum = torch.sum(x * weights, dim=1)
        return weighted_sum, weights


class MultimodalFusionModel(nn.Module):
    """
    Multimodal fusion using SAME visual backbone as Task 1/2.
    
    Visual branch = EngagementLSTM backbone (LSTM 32, 2 layers)
    rPPG branch = simple MLP (14 → 16)
    Fusion = concat (32 + 16 = 48) → FC(32) → heads
    """
    
    def __init__(self, visual_dim=9, rppg_dim=14, lstm_hidden=32,
                 lstm_layers=2, dropout=0.3, num_classes=4):
        super().__init__()
        
        # ---- Visual Branch (SAME as Task 1/2 EngagementLSTM) ----
        self.visual_lstm = nn.LSTM(
            input_size=visual_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.visual_attention = Attention(lstm_hidden)
        self.visual_fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        visual_out_dim = 32
        
        # ---- rPPG Branch (simple MLP) ----
        self.rppg_branch = nn.Sequential(
            nn.Linear(rppg_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        rppg_out_dim = 16
        
        # ---- Fusion + Classifier ----
        fused_dim = visual_out_dim + rppg_out_dim  # 32 + 16 = 48
        
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Task 5a: Binary (Engaged vs Not Engaged)
        self.head_binary = nn.Linear(32, 1)
        
        # Task 5b: Multi-class (4 levels)
        self.head_multi = nn.Linear(32, num_classes)
        
        # Regression head
        self.head_reg = nn.Linear(32, 1)
    
    def forward(self, visual_seq, rppg_feat):
        """
        Args:
            visual_seq: (B, T, 9) - temporal visual features
            rppg_feat:  (B, 14) - per-video rPPG features
        """
        # Visual branch (same ops as EngagementLSTM)
        lstm_out, _ = self.visual_lstm(visual_seq)
        visual_ctx, _ = self.visual_attention(lstm_out)  # (B, 32)
        visual_feat = self.visual_fc(visual_ctx)  # (B, 32)
        
        # rPPG branch
        rppg_out = self.rppg_branch(rppg_feat)  # (B, 16)
        
        # Early fusion: concatenate
        fused = torch.cat([visual_feat, rppg_out], dim=1)  # (B, 48)
        
        # Classify
        features = self.classifier(fused)
        
        out_bin = self.head_binary(features)
        out_multi = self.head_multi(features)
        out_reg = self.head_reg(features)
        
        return out_bin, out_multi, out_reg
    
    def load_task12_weights(self, checkpoint_path):
        """
        Load pretrained Task 1/2 weights for the visual branch.
        Maps EngagementLSTM weights → our visual branch.
        """
        try:
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            
            mapping = {
                'lstm.weight_ih_l0': 'visual_lstm.weight_ih_l0',
                'lstm.weight_hh_l0': 'visual_lstm.weight_hh_l0',
                'lstm.bias_ih_l0': 'visual_lstm.bias_ih_l0',
                'lstm.bias_hh_l0': 'visual_lstm.bias_hh_l0',
                'lstm.weight_ih_l1': 'visual_lstm.weight_ih_l1',
                'lstm.weight_hh_l1': 'visual_lstm.weight_hh_l1',
                'lstm.bias_ih_l1': 'visual_lstm.bias_ih_l1',
                'lstm.bias_hh_l1': 'visual_lstm.bias_hh_l1',
                'attention.attn.weight': 'visual_attention.attn.weight',
                'attention.attn.bias': 'visual_attention.attn.bias',
                'fc.0.weight': 'visual_fc.0.weight',
                'fc.0.bias': 'visual_fc.0.bias',
            }
            
            new_state = self.state_dict()
            loaded = 0
            for old_key, new_key in mapping.items():
                if old_key in state and new_key in new_state:
                    if state[old_key].shape == new_state[new_key].shape:
                        new_state[new_key] = state[old_key]
                        loaded += 1
            
            self.load_state_dict(new_state)
            print(f"    Loaded {loaded}/{len(mapping)} Task 1/2 weights for visual branch")
            return loaded > 0
        except Exception as e:
            print(f"    Could not load Task 1/2 weights: {e}")
            return False


class VisualOnlyModel(nn.Module):
    """
    Visual-only baseline for ablation.
    EXACT same architecture as Task 1/2 EngagementLSTM + multi-class head.
    """
    
    def __init__(self, visual_dim=9, hidden_dim=32, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(visual_dim, hidden_dim, num_layers,
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_binary = nn.Linear(32, 1)
        self.head_multi = nn.Linear(32, 4)
        self.head_reg = nn.Linear(32, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        ctx, _ = self.attention(out)
        feat = self.fc(ctx)
        return self.head_binary(feat), self.head_multi(feat), self.head_reg(feat)
    
    def load_task12_weights(self, checkpoint_path):
        """Load pretrained Task 1/2 weights."""
        try:
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            own_state = self.state_dict()
            loaded = 0
            for key in state:
                if key in own_state and state[key].shape == own_state[key].shape:
                    own_state[key] = state[key]
                    loaded += 1
            self.load_state_dict(own_state)
            print(f"    Loaded {loaded} Task 1/2 weights for visual-only model")
            return loaded > 0
        except Exception as e:
            print(f"    Could not load Task 1/2 weights: {e}")
            return False
