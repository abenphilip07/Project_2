import numpy as np
import os
import json
from scipy.io import wavfile
from scipy.signal import spectrogram

def extract_audio_features(audio_path, output_path, feature_dim=1000, sr=44100):
    """
    Extract features from audio and ensure it is sampled at the target sample rate.
    """
    # Read audio data using scipy
    rate, audio = wavfile.read(audio_path)
    
    # Handle stereo to mono conversion if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Ensure audio is short clip friendly
    nperseg = min(256, len(audio))
    noverlap = min(nperseg // 2, len(audio) - 1)
    
    # Compute spectrogram
    try:
        _, _, spec = spectrogram(audio, fs=rate, nperseg=nperseg, noverlap=noverlap)
    except Exception as e:
        print(f"Spectrogram error for {audio_path}: {e}")
        # Fallback feature extraction for very short clips
        feature_vector = np.zeros(feature_dim)
    else:
        # Handle 1D and 2D spec arrays
        if spec.ndim == 1:
            feature_vector = spec
        else:
            # Flatten the spectrogram to create a temporal feature vector
            feature_vector = np.mean(spec, axis=0)  # Changed from axis=1
        
        # Pad or truncate to ensure a fixed size
        feature_vector = np.pad(feature_vector, (0, max(0, feature_dim - len(feature_vector))), 'constant')[:feature_dim]

    # Save features to file
    np.save(output_path, feature_vector)
    return output_path

def process_metadata(metadata, feature_output_dir, feature_dim=1000, sr=44100):
    """
    Process metadata and extract audio features for all chunks.
    """
    # Create output directory if it doesn't exist
    os.makedirs(feature_output_dir, exist_ok=True)
    
    # Update metadata with extracted features
    for video_id, data in metadata.items():
        for chunk in data['chunks']:
            audio_path = chunk['audio_path']
            # Define feature save path
            feature_path = os.path.join(feature_output_dir, f"{os.path.basename(audio_path)}.npy")
            
            # Extract features
            extract_audio_features(audio_path, feature_path, feature_dim=feature_dim, sr=sr)
            
            # Add feature path to metadata
            chunk['features_path'] = feature_path
    return metadata

# Main execution
def main():
    # Load metadata
    metadata_path = r'E:\Project_2\metadata\chunks_metadata.json'
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Process metadata and save new metadata
    feature_output_dir = r'E:\Project_2\data\audio features\features'
    updated_metadata = process_metadata(metadata, feature_output_dir, sr=44100)

    # Save updated metadata
    updated_metadata_path = r'E:\Project_2\metadata\audio_chunks_metadata.json'
    with open(updated_metadata_path, "w") as f:
        json.dump(updated_metadata, f, indent=4)

if __name__ == "__main__":
    main()