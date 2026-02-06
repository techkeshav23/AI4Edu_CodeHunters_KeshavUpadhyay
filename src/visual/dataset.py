import os
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from src.common.config import Config

class EngagementDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, phase='train'):
        """
        Args:
            video_paths (list): List of paths to video files or frame directories.
            labels (list): List of labels corresponding to videos.
            transform (callable, optional): PyTorch transforms for augmentation.
            phase (str): 'train' or 'val'.
        """
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # In a real scenario, avoiding reading full video on the fly is better.
        # We usually pre-extract frames to disk. 
        # Here we assume video_path points to a folder of extracted face frames.
        
        frame_dir = self.video_paths[idx]
        label = self.labels[idx]
        
        # Binary Mapping for Task 1 (if raw labels are 0, 0.33, 0.66, 1)
        if Config.NUM_CLASSES == 2:
            # 0 and 0.33 -> 0 (Low)
            # 0.66 and 1 -> 1 (High)
            if label <= 0.33:
                target = 0
            else:
                target = 1
        else:
            # Task 2 direct mapping
            # We might need a dict to map 0.33 -> 1, 0.66 -> 2 etc if using CrossEntropy
            target = label 

        # Load a sample frame (middle frame or random frame strategy)
        # We handle cases where no frames were extracted (e.g. face not detected)
        frames = []
        if os.path.exists(frame_dir):
            frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
            
        if frames:
            # Sampling strategy: Take middle frame
            # In future we can take random or multiple frames
            mid_idx = len(frames) // 2
            img_path = os.path.join(frame_dir, frames[mid_idx])
            try:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                # Corrupt image fallback
                print(f"Error reading image {img_path}: {e}")
                image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Fallback for empty/error video (Black Image)
            # This ensures training doesn't crash on bad data
            image = np.zeros((224, 224, 3), dtype=np.uint8) 
            # Note: We return numpy array, transform will convert to tensor

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(target, dtype=torch.long)
