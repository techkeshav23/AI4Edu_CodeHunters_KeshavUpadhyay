import cv2
import torch
import numpy as np
import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings (Clean Terminal)
os.environ["GLOG_minloglevel"] = "3"  # Suppress MediaPipe logs
from torchvision import transforms
from PIL import Image
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

# ----------------- CONFIGURATION -----------------
# Define Config inline to avoid dependency issues
class Config:
    IMG_SIZE = 224
    # Updated for Binary Model (Detected from Error)
    # SWAPPED LABELS based on user feedback (Student looking at screen was labeled 1/Distracted)
    # Therefore: 1 must be Engaged, 0 must be Distracted
    LABELS_MAP = {
        0: "Distracted / Not Focused",
        1: "Engaged / Focused"
    }
    # Colors for graph
    COLORS = {
        0: (0, 0, 255),    # Red (Bad)
        1: (0, 255, 0)     # Green (Good)
    }

# ----------------- MODEL LOADER -----------------
class VisualModel(torch.nn.Module):
    def __init__(self, num_classes=2): # Changed default to 2
        super(VisualModel, self).__init__()
        from torchvision import models
        # Load architecture (must match training)
        self.backbone = models.resnet18(weights=None) 
        num_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def load_model(model_path, device):
    print(f"Loading model from: {model_path}")
    # Try initializing with 2 classes first (since error said [2, 512])
    model = VisualModel(num_classes=2)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("Success: Loaded Binary Model (2 Classes)")
    except RuntimeError as e:
        print("Mismatch detected, trying 4 classes...")
        model = VisualModel(num_classes=4)
        model.load_state_dict(checkpoint)
        print("Success: Loaded Multi-Class Model (4 Classes)")
    
    model.to(device)
    model.eval()
    return model

# ----------------- FACE EXTRACTOR -----------------
class LiveFaceExtractor:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def get_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            # Get largest face
            detection = results.detections[0] # Assume first is primary
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Padding MUST match training (face_extractor.py used padding=0)
            padding = 0
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(iw - x, w + 2*padding)
            h = min(ih - y, h + 2*padding)
            
            face = frame[y:y+h, x:x+w]
            # Draw rect on original frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            return face, frame
        return None, frame

# ----------------- MAIN DEMO -----------------
def main():
    parser = argparse.ArgumentParser(description="Run Analyzer Demo")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--model", type=str, default="models/visual/best_visual_model.pth", help="Path to .pth model file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Load Model
    model = load_model(args.model, device)
    
    # Face Extractor
    extractor = LiveFaceExtractor()
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Video Capture
    cap = cv2.VideoCapture(args.video)
    
    # Graph Setup
    history_len = 50
    engagement_history = [0] * history_len
    fig, ax = plt.subplots(figsize=(5, 2), dpi=100)
    
    print("Starting Demo... Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Loop video: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

        # 1. Detect Face
        face_img, display_frame = extractor.get_face(frame)
        
        label_text = "No Face"
        color = (100, 100, 100)
        
        if face_img is not None and face_img.size > 0:
            try:
                # 2. Prepare for Model
                pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_img).unsqueeze(0).to(device)
                
                # 3. Predict
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_idx].item()
                
                # 4. Map Result
                label_text = f"{Config.LABELS_MAP[pred_idx]} ({confidence*100:.1f}%)"
                color = Config.COLORS[pred_idx]
                
                # Update Graph Data for BINARY model
                # pred_idx=1 (Engaged) -> graph goes UP (1)
                # pred_idx=0 (Distracted) -> graph goes DOWN (0)
                graph_val = pred_idx
                engagement_history.append(graph_val)
                engagement_history.pop(0)
                
            except Exception as e:
                print(f"Frame Error: {e}")
        else:
             engagement_history.append(0) # Drop to 0 if no face
             engagement_history.pop(0)

        # ----------------- VISUALIZATION -----------------
        # Draw Plot
        ax.clear()
        ax.set_ylim(-0.2, 1.2)  # Binary: 0 to 1 range
        ax.fill_between(range(len(engagement_history)), engagement_history, alpha=0.3, color='lime')
        ax.plot(engagement_history, color='cyan', linewidth=2)
        ax.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5)  # Threshold line
        ax.set_title("Engagement Level", color='white', fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Distracted', 'Engaged'], color='white', fontsize=8)
        ax.tick_params(axis='x', colors='white')
        fig.patch.set_facecolor('black')
        ax.patch.set_facecolor('black')
        
        # Convert Plot to Image
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        # raw_data = renderer.tostring_rgb() # Old method failing
        
        # New robust method for Matplotlib 3.8+
        s, (width, height) = canvas.print_to_buffer()
        plot_img = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        
        # Convert RGBA to BGR
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        
        # Overlay Plot on Frame (Bottom Right)
        h, w, _ = display_frame.shape
        ph, pw, _ = plot_img.shape
        
        # Resize plot if too big
        if pw > w: 
            scale = w / pw
            plot_img = cv2.resize(plot_img, (0,0), fx=scale, fy=scale)
            ph, pw, _ = plot_img.shape
            
        display_frame[h-ph:h, w-pw:w] = plot_img
        
        # Draw Label Text
        cv2.putText(display_frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        # Show Grid
        cv2.imshow('Analyzer Demo', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()