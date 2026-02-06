import os

class Config:
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RAW_DIR = os.path.join(DATA_DIR, 'raw')
    # Updated: Videos are inside a 'Train' subfolder on the server
    RAW_VIDEO_DIR = os.path.join(RAW_DIR, 'videos', 'Train')
    LABELS_PATH = os.path.join(RAW_DIR, 'labels_train.xlsx')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    
    # Visual Model Hyperparameters
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    
    # Task Settings
    # Task 1: Binary (0, 1) -> num_classes = 2
    # Task 2: Multi (0, 0.33, 0.66, 1) -> num_classes = 4
    NUM_CLASSES = 2 
    
    # MediaPipe Settings
    CONFIDENCE_THRESHOLD = 0.5
