import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.config import Config
from src.visual.dataset import EngagementDataset
from src.visual.models import VisualEngagementModel

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Transforms (Augmentation)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Labels
    df = pd.read_excel(Config.LABELS_PATH)
    video_names = df.iloc[:, 0].astype(str).tolist() # Assuming col 0 is name
    labels = df.iloc[:, 1].tolist()                  # Assuming col 1 is label
    
    # Map video names to processed folders
    processed_paths = []
    valid_labels = []
    
    for name, lbl in zip(video_names, labels):
        # Handle extension removal for folder name
        folder_name = os.path.splitext(name)[0]
        folder_path = os.path.join(Config.PROCESSED_DIR, folder_name)
        
        if os.path.exists(folder_path) and len(os.listdir(folder_path)) > 0:
            processed_paths.append(folder_path)
            valid_labels.append(lbl)
            
    print(f"Found {len(processed_paths)} valid processed videos out of {len(df)}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        processed_paths, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels
    )
    
    # Save Validation List for User Verification
    val_list_path = os.path.join(Config.PROJECT_ROOT, 'validation_split.txt')
    with open(val_list_path, 'w') as f:
        f.write("Videos used for Validation (Testing):\n")
        for path in X_val:
            # Extract video name from folder path
            vid_name = os.path.basename(path)
            f.write(f"{vid_name}\n")
    print(f"saved list of {len(X_val)} validation videos to: {val_list_path}")
    
    # Datasets & Loaders
    train_dataset = EngagementDataset(X_train, y_train, transform=train_transform, phase='train')
    val_dataset = EngagementDataset(X_val, y_val, transform=val_transform, phase='val')
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model Setup
    model = VisualEngagementModel(num_classes=Config.NUM_CLASSES).to(device)
    
    # Improvement (Weighted Loss): Handle class imbalance (30 vs 44 videos)
    # We give more weight to Class 0 (Minority/Distracted) approx 1.5x
    # Class 1 (Majority/Engaged) gets 1.0x
    class_weights = torch.tensor([1.5, 1.0]).to(device)
    
    # Combined with Label Smoothing for Noisy Labels
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training Loop
    best_acc = 0.0
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(Config.PROJECT_ROOT, 'models', 'best_visual_model.pth')
            # Ensure folder exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model to {save_path}")

if __name__ == "__main__":
    main()
