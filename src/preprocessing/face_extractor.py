import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm

class FaceExtractor:
    def __init__(self, confidence=0.5):
        # Direct approach that usually works across versions
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=confidence)

    def process_video(self, video_path, output_folder, sample_rate=1):
        """
        Reads a video, detects faces, crops them, and saves to output_folder.
        sample_rate: Process every Nth frame (e.g., 1=every frame, 5=every 5th frame)
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use filename as prefix
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        with tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frame") as pbar:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                frame_count += 1
                pbar.update(1)
                
                # Skip frames based on sample rate
                if frame_count % sample_rate != 0:
                    continue
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_frame)
                
                if results.detections:
                    for detection in results.detections:
                        # Get bounding box
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                     int(bboxC.width * iw), int(bboxC.height * ih)
                        
                        # Add some margin (padding)
                        # We keep padding tight to focus on facial features (eyes/gaze) 
                        # as per "Improvement B" feedback
                        padding = 0 
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(iw - x, w + 2 * padding)
                        h = min(ih - y, h + 2 * padding)
                        
                        # Crop face
                        face_crop = frame[y:y+h, x:x+w]
                        
                        if face_crop.size > 0:
                            save_path = os.path.join(output_folder, f"{video_name}_frame_{frame_count:04d}.jpg")
                            cv2.imwrite(save_path, face_crop)
                            saved_count += 1
                        
                        # We assume 1 student per video, so break after first face
                        break
                        
        cap.release()
        return saved_count
