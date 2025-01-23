import os
import json
import csv
from tqdm import tqdm
import numpy as np
from moviepy.editor import VideoFileClip
import scipy.io.wavfile as wav

import pandas as pd

def separate_audio_video_with_chunking(chunks_folder, metadata_path, output_folder, chunk_size=30, overlap=5):
    """Separate audio and video from chunks with updated metadata structure."""
    import os
    import json
    from tqdm import tqdm
    from moviepy.video.io.VideoFileClip import VideoFileClip
    import numpy as np
    import scipy.io.wavfile as wav
    
    # Create output directories
    video_output_dir = os.path.join(output_folder, "videos")
    audio_output_dir = os.path.join(output_folder, "audio")
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)
    
    # Load metadata CSV
    metadata = pd.read_csv(metadata_path)
    
    # Prepare new metadata list
    new_metadata = []
    
    # Process each video
    print("Processing videos with chunking...")
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        try:
            video_name = row["path"]  # Relative video name
            video_full_path = row["full_path"]  # Full video path
            
            if not os.path.exists(video_full_path):
                print(f"Warning: Video not found - {video_full_path}")
                continue
            
            # Load video
            video = VideoFileClip(video_full_path)
            duration = video.duration  # Duration in seconds
            fps = video.fps
            
            # Calculate chunks
            total_frames = int(duration * fps)
            chunk_frames = chunk_size * fps
            overlap_frames = overlap * fps
            
            frame_starts = list(range(0, total_frames, int(chunk_frames - overlap_frames)))
            
            for i, start_frame in enumerate(frame_starts):
                end_frame = min(start_frame + chunk_frames, total_frames)
                
                # Extract video chunk
                chunk_video = video.subclip(start_frame / fps, end_frame / fps)
                chunk_video_name = f"{os.path.splitext(video_name)[0]}_chunk{i:03d}_video.mp4"
                chunk_video_path = os.path.join(video_output_dir, chunk_video_name)
                chunk_video.write_videofile(chunk_video_path, audio=False, logger=None)
                
                # Extract audio chunk
                chunk_audio_name = f"{os.path.splitext(video_name)[0]}_chunk{i:03d}_audio.wav"
                chunk_audio_path = os.path.join(audio_output_dir, chunk_audio_name)
                if chunk_video.audio is not None:
                    chunk_video.audio.write_audiofile(chunk_audio_path, logger=None)
                else:
                    print(f"No audio found in chunk: {chunk_video_name}. Creating silent audio.")
                    duration_chunk = chunk_video.duration
                    sample_rate = 44100
                    silent_audio = np.zeros(int(duration_chunk * sample_rate))
                    wav.write(chunk_audio_path, sample_rate, silent_audio.astype(np.float32))
                
                # Append metadata
                new_metadata.append({
                    "filename": chunk_video_name,
                    "audio_path": chunk_audio_name,
                    "original": row["source"],
                    "chunk_index": i,
                    "label": row["category"],
                    "split": row["type"]
                })
                
            # Close video
            video.close()
        
        except Exception as e:
            print(f"Error processing video {video_name}: {str(e)}")
    
    # Save new metadata
    new_metadata_path = os.path.join(output_folder, "chunked_metadata.json")
    with open(new_metadata_path, "w") as f:
        json.dump(new_metadata, f, indent=4)
    
    print("\nProcessing completed!")
    print(f"Videos saved to: {video_output_dir}")
    print(f"Audio saved to: {audio_output_dir}")
    print(f"New metadata saved to: {new_metadata_path}")
    
    return new_metadata


def process_video_with_chunking(video_name, video_path, label, split, video_output_dir, audio_output_dir, chunk_size, overlap):
    """Process video into chunks and extract audio/video."""
    try:
        video = VideoFileClip(video_path)
        fps = video.fps
        total_frames = int(video.duration * fps)
        chunk_step = chunk_size - overlap
        results = []
        
        for start_frame in range(0, total_frames, chunk_step * fps):
            end_frame = min(start_frame + chunk_size * fps, total_frames)
            
            # Define chunked video path
            base_name = os.path.splitext(video_name)[0]
            chunk_index = start_frame // (chunk_step * fps)
            chunk_video_filename = f"{base_name}_chunk{chunk_index}_video.mp4"
            chunk_audio_filename = f"{base_name}_chunk{chunk_index}_audio.wav"
            
            # Video chunking
            chunk_video_path = os.path.join(video_output_dir, chunk_video_filename)
            video.subclip(start_frame / fps, end_frame / fps).write_videofile(chunk_video_path, audio=False, logger=None)
            
            # Audio extraction
            chunk_audio_path = os.path.join(audio_output_dir, chunk_audio_filename)
            if video.audio is not None:
                video.audio.subclip(start_frame / fps, end_frame / fps).write_audiofile(chunk_audio_path, logger=None)
            else:
                print(f"No audio found in video chunk: {chunk_video_filename}. Creating silent audio file.")
                duration = (end_frame - start_frame) / fps
                sample_rate = 44100
                silent_audio = np.zeros(int(duration * sample_rate))
                wav.write(chunk_audio_path, sample_rate, silent_audio.astype(np.float32))
            
            # Create metadata entry
            result = {
                'filename': chunk_video_filename,
                'audio_path': chunk_audio_path,
                'label': label,
                'split': split,
                'chunk_index': chunk_index
            }
            results.append(result)
        
        video.close()
        return results

    except Exception as e:
        print(f"Error processing video {video_name}: {str(e)}")
        return None


# Usage
dataset_folder = r"E:\Project_2\data\FakeAVCeleb_Sampled"
metadata_path = r"E:\Project_2\data\FakeAVCeleb_Sampled\metadata_updated.csv"
output_folder = r"E:\Project_2\data\Processed_Chunks"

new_metadata = separate_audio_video_with_chunking(
    dataset_folder, 
    metadata_path, 
    output_folder,
    chunk_size=30,
    overlap=5
)
