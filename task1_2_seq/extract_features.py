import argparse
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# Constants
FPS_TARGET = 6  # Extract 6 frames per second
MP_FACE_MESH = mp.solutions.face_mesh

def get_head_pose(landmarks, shape):
    img_h, img_w, _ = shape
    face_3d = []
    face_2d = []

    # Nose, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth
    # Indices: Nose: 1, Chin: 199, Left Eye: 33, Right Eye: 263, Left Mouth: 61, Right Mouth: 291
    # Choosing specific landmarks for PnP
    key_landmarks = [1, 199, 33, 263, 61, 291]

    for idx, lm in enumerate(landmarks):
        if idx in key_landmarks:
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z]) # Depth approximation

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Camera internals
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])

    # Dist matrix
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
    if not success:
        return 0, 0, 0

    # Get Rot Matrix
    rmat, jac = cv2.Rodrigues(rot_vec)

    # Get Angles (handle different OpenCV versions)
    result = cv2.RQDecomp3x3(rmat)
    angles = result[0]  # First element is always the angles array

    # Pitch, Yaw, Roll
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    return x, y, z # Pitch, Yaw, Roll

def get_eye_ratio(landmarks, eye_indices):
    # Eye Aspect Ratioish
    # Vertical distance / Horizontal distance
    # Not implementing full EAR for simplicity, just mean eye opening
    # Top/Bottom landmarks
    top = landmarks[eye_indices[1]] # 159 for left
    bottom = landmarks[eye_indices[3]] # 145 for left
    
    # Left/Right landmarks
    left = landmarks[eye_indices[0]] # 33
    right = landmarks[eye_indices[2]] # 133
    
    v_dist = np.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
    h_dist = np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2)
    
    if h_dist == 0: return 0
    return v_dist / h_dist

def process_video(video_path, face_mesh):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Default
    
    frame_interval = int(round(fps / FPS_TARGET))
    if frame_interval < 1: frame_interval = 1
    
    features = []
    frame_count = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        if frame_count % frame_interval == 0:
            # Convert
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image.shape
            
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lm = face_landmarks.landmark
                    
                    # 1. Head Pose (3)
                    pitch, yaw, roll = get_head_pose(lm, (img_h, img_w, 3))
                    
                    # 2. Eye Gaze proxies (Iris/Pupil are hard without Iris model, using eye centers)
                    # Left Eye Center (approx)
                    # 33: inner, 133: outer. Center ~ avg
                    le_x = (lm[33].x + lm[133].x) / 2
                    le_y = (lm[33].y + lm[133].y) / 2
                    
                    # Right Eye Center
                    # 362: inner, 263: outer
                    re_x = (lm[362].x + lm[263].x) / 2
                    re_y = (lm[362].y + lm[263].y) / 2
                    
                    # 3. Blink (EAR) (2)
                    # Left Eye indices: 33, 159, 133, 145 (Left, Top, Right, Bottom)
                    left_ear = get_eye_ratio(lm, [33, 159, 133, 145])
                    # Right Eye indices: 362, 386, 263, 374
                    right_ear = get_eye_ratio(lm, [362, 386, 263, 374])
                    
                    # Vector: [Pitch, Yaw, Roll, LE_X, LE_Y, RE_X, RE_Y, L_EAR, R_EAR]
                    feat = [pitch, yaw, roll, le_x, le_y, re_x, re_y, left_ear, right_ear]
                    features.append(feat)
                    break # Only first face
            else:
                # No face found, append zeros or last known?
                # For safety, append zeros (model learns "no face" = distracted)
                features.append([0]*9)
        
        frame_count += 1
        
    cap.release()
    return np.array(features) # (T, 9)

def process_video_from_frames(folder_path, face_mesh):
    # Get all images
    frames = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
    
    # We assume the frames in processed folder are already sampled or we take all
    # Depending on density. If 100 frames total, that's enough.
    
    features = []
    
    for img_path in frames:
        image = cv2.imread(img_path)
        if image is None: continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = image.shape
        
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = face_landmarks.landmark
                
                # 1. Head Pose (3)
                pitch, yaw, roll = get_head_pose(lm, (img_h, img_w, 3))
                
                # 2. Eye Gaze (4)
                le_x = (lm[33].x + lm[133].x) / 2
                le_y = (lm[33].y + lm[133].y) / 2
                re_x = (lm[362].x + lm[263].x) / 2
                re_y = (lm[362].y + lm[263].y) / 2
                
                # 3. Blink (2)
                left_ear = get_eye_ratio(lm, [33, 159, 133, 145])
                right_ear = get_eye_ratio(lm, [362, 386, 263, 374])
                
                feat = [pitch, yaw, roll, le_x, le_y, re_x, re_y, left_ear, right_ear]
                features.append(feat)
                break 
        else:
            features.append([0]*9)

    return np.array(features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True, help='Path to video folder OR processed frames folder')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels excel')
    parser.add_argument('--output_dir', type=str, default='dataset/features', help='Where to save .npy files')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Init MediaPipe
    with MP_FACE_MESH.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        df = pd.read_excel(args.labels)
        video_names = df.iloc[:, 0].astype(str).tolist()
        
        print(f"Processing {len(video_names)} videos...")
        
        for name in tqdm(video_names):
            clean_name = os.path.splitext(name)[0]
            
            # 1. Check for FOLDER (Processed Frames)
            folder_p = os.path.join(args.video_dir, clean_name)
            
            # 2. Check for FILE (Raw Video)
            vid_path = None
            if os.path.isdir(folder_p):
                 vid_path = folder_p
                 is_video_file = False
            else:
                for ext in ['.avi', '.mp4', '.mov']:
                    p = os.path.join(args.video_dir, clean_name + ext)
                    if os.path.exists(p):
                        vid_path = p
                        is_video_file = True
                        break
            
            if not vid_path:
                # Fallback: maybe the folder doesn't have the filename without extension?
                # Sometimes processed folders are just names.
                # Let's try to match partials or user might have passed 'data/processed'
                print(f"Warning: Data for {clean_name} not found in {args.video_dir}")
                continue
            
            npy_path = os.path.join(args.output_dir, clean_name + '.npy')
            if os.path.exists(npy_path):
                continue 
            
            try:
                if is_video_file:
                    features = process_video(vid_path, face_mesh)
                else:
                    features = process_video_from_frames(vid_path, face_mesh)
                    
                if features is not None and len(features) > 0:
                    np.save(npy_path, features)
                else:
                    print(f"Empty features for {name}")
            except Exception as e:
                print(f"Error processing {name}: {e}")

if __name__ == "__main__":
    main()
