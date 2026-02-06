import os
import sys
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.config import Config
from src.preprocessing.face_extractor import FaceExtractor

def main():
    print("Starting Data Preparation on Server...")
    print(f"Reading labels from: {Config.LABELS_PATH}")
    
    # 1. Load Excel
    try:
        df = pd.read_excel(Config.LABELS_PATH)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Check columns
    # Adjust column names based on your actual excel file
    # Assuming columns like 'video_name' and 'label' or similar
    # If standard dataset, it might be just filenames
    print(f"Columns found: {df.columns.tolist()}")
    
    # Initialize Extractor
    extractor = FaceExtractor(confidence=Config.CONFIDENCE_THRESHOLD)
    
    processed_count = 0
    
    # 2. Iterate and Process
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Videos"):
        # We assume first column works as ID/Name if not explicitly named 'video'
        video_name = str(row.iloc[0]) 
        
        # SMART FIX: Check if extension is already present trying to match server file
        # Common video extensions
        valid_exts = ('.mp4', '.avi', '.mov', '.webm', '.wmv', '.MP4', '.AVI')
        
        # Logic:
        # 1. Try exact name from Excel
        # 2. If not found, try appending extensions
        # 3. Handle double extension case (e.g. .MP4.mp4)
        
        candidate_names = [video_name] # Default: use as is
        
        # If no extension, try adding them
        if not video_name.lower().endswith(valid_exts):
             candidate_names = [video_name + ext for ext in valid_exts]
        
        # Try to find the file
        found_video_path = None
        for cand in candidate_names:
            p = os.path.join(Config.RAW_VIDEO_DIR, cand)
            if os.path.exists(p):
                found_video_path = p
                video_name = cand # Update to correct name
                break
        
        if not found_video_path:
             # Try case insensitive match if still not found
             try:
                 server_files = os.listdir(Config.RAW_VIDEO_DIR)
                 # Check if any server file starts with video_name
                 # (simplistic approximation)
                 pass
             except:
                 pass
                 
             # Fallback to default behavior but log warning
             video_path = os.path.join(Config.RAW_VIDEO_DIR, video_name if video_name.lower().endswith(valid_exts) else video_name + '.mp4')
        else:
            video_path = found_video_path

        
        # Output folder for this specific video's frames
        # e.g. data/processed/video_01/
        # Using name without extension for folder
        folder_name = os.path.splitext(video_name)[0]
        output_folder = os.path.join(Config.PROCESSED_DIR, folder_name)
        
        # Check if already processed (Smart Resume)
        if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 50:
             # If folder exists and has frames, assume done.
             # print(f"Skipping {video_name} (Already processed)")
             continue

        if not os.path.exists(video_path):
            print(f"Warning: Video not found at {video_path}")
            continue
        
        if os.path.exists(video_path):
            count = extractor.process_video(video_path, output_folder, sample_rate=5) # Reduced rate for speed
            if count > 0:
                processed_count += 1
        else:
            print(f"Warning: Video not found at {video_path}")

    print(f"Preprocessing Complete. Processed {processed_count}/{len(df)} videos.")

if __name__ == "__main__":
    main()
