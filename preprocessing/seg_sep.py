import os
import cv2
import pandas as pd
import json
from pydub import AudioSegment
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio

# Input parameters
input_csv_path = r'E:\Project_2\data\FakeAVCeleb_Sampled\metadata_updated1.csv'
output_dir = r'E:\Project_2\data\FakeAVCeleb_Sampled\chunks'
output_metadata_path = os.path.join(output_dir, "metadata.json")
chunk_size = 30
overlap = 5

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to process each video
def process_video(row, chunk_size, overlap, output_dir):
    video_path = row['full_path']
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    
    video_cap = cv2.VideoCapture(video_path)
    frame_rate = int(video_cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_chunks = (total_frames - overlap) // (chunk_size - overlap)
    
    # Create subdirectories for video and audio
    video_chunk_dir = os.path.join(output_dir, video_name, "video")
    audio_chunk_dir = os.path.join(output_dir, video_name, "audio")
    os.makedirs(video_chunk_dir, exist_ok=True)
    os.makedirs(audio_chunk_dir, exist_ok=True)

    # Extract audio from video
    audio_path = os.path.join(audio_chunk_dir, f"{video_name}.wav")
    ffmpeg_extract_audio(video_path, audio_path)
    
    # Metadata for the current video
    video_metadata = {
        "original_video": video_path,
        "total_chunks": total_chunks,
        "chunks": []
    }
    
    chunk_index = 0
    for start_frame in range(0, total_frames, chunk_size - overlap):
        if start_frame + chunk_size > total_frames:
            break  # Skip incomplete chunks
        
        # Create chunk file paths
        chunk_video_path = os.path.join(video_chunk_dir, f"{video_name}_chunk{chunk_index:03d}.mp4")
        chunk_audio_path = os.path.join(audio_chunk_dir, f"{video_name}_chunk{chunk_index:03d}.wav")
        
        # Write video chunk
        video_writer = None
        frame_idx = 0
        while frame_idx < chunk_size:
            ret, frame = video_cap.read()
            if not ret:
                break
            
            if video_writer is None:
                height, width, _ = frame.shape
                video_writer = cv2.VideoWriter(
                    chunk_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    frame_rate,
                    (width, height)
                )
            
            video_writer.write(frame)
            frame_idx += 1

        if video_writer:
            video_writer.release()
        
        # Write audio chunk
        audio = AudioSegment.from_file(audio_path)
        start_time = (start_frame / frame_rate) * 1000  # Convert to milliseconds
        end_time = ((start_frame + chunk_size) / frame_rate) * 1000
        chunk_audio = audio[start_time:end_time]
        chunk_audio.export(chunk_audio_path, format="wav")
        
        # Update metadata
        video_metadata["chunks"].append({
            "chunk_index": chunk_index,
            "audio_path": chunk_audio_path,
            "video_path": chunk_video_path
        })
        
        chunk_index += 1
    
    video_cap.release()
    return video_metadata

# Read metadata CSV
df = pd.read_csv(input_csv_path)

# Main metadata structure
all_metadata = {}

# Process each video and collect metadata
for _, row in df.iterrows():
    video_metadata = process_video(row, chunk_size, overlap, output_dir)
    all_metadata[row['filename']] = video_metadata

# Save the consolidated metadata JSON
with open(output_metadata_path, 'w') as metadata_file:
    json.dump(all_metadata, metadata_file, indent=4)

print(f"Processing complete. Metadata saved to {output_metadata_path}")
