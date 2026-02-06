import torch
import torch.nn as nn
from torchvision import models

class VisualEngagementModel(nn.Module):
    def __init__(self, num_classes=2, model_name='resnet18', pretrained=True):
        """
        Standard CNN for Visual Engagement Classification.
        Supports binary (Task 1) and multi-class (Task 2).
        """
        super(VisualEngagementModel, self).__init__()
        
        # Load a standard backbone (ResNet18 is a good balance of speed/accuracy)
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.fc.in_features
        elif model_name == 'efficientnet':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Model {model_name} not supported")

        # Replace the final classification layer
        # For ResNet, it's 'fc'. For EfficientNet, it's 'classifier'.
        if model_name == 'resnet18':
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif model_name == 'efficientnet':
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
            
    def forward(self, x):
        # Input shape: (Batch, Channels, Height, Width)
        return self.backbone(x)
