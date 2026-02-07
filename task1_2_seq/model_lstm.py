import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: [Batch, Seq, Hidden]
        # weights: [Batch, Seq, 1]
        weights = torch.softmax(self.attn(x), dim=1)
        # weighted_sum: [Batch, Hidden]
        weighted_sum = torch.sum(x * weights, dim=1)
        return weighted_sum, weights

class EngagementLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=32, num_layers=2):
        super(EngagementLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Binary Classification (Task 1)
        # Output: Logits (will use BCEWithLogitsLoss)
        self.head_binary = nn.Linear(32, 1)
        
        # Regression (Task 2)
        # Output: Score 0-1 (Sigmoid to force range) or Raw
        # Plan says Linear activation, let's use Linear and Clamp/Loss handles it
        self.head_regression = nn.Linear(32, 1)

    def forward(self, x):
        # x: [Batch, Seq, Feature]
        out, _ = self.lstm(x)
        
        # Attention Pooling
        context, attn_weights = self.attention(out)
        
        # Shared Dense
        feat = self.fc(context)
        
        # Heads
        out_binary = self.head_binary(feat)
        out_reg = self.head_regression(feat)
        
        return out_binary, out_reg

