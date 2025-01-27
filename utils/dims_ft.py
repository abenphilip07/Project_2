import os
import json
import numpy as np

def diagnose_feature_extraction(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    feature_shapes = []
    problematic_files = []

    for video_name, video_data in metadata.items():
        for chunk in video_data["chunks"]:
            feature_path = chunk["reduced_audio_feature_path"]
            try:
                features = np.load(feature_path)
                feature_shapes.append(features.shape)
                print(f"Video {video_name}, Chunk {chunk['chunk_index']}: Shape {features.shape}")
            except Exception as e:
                print(f"Error loading {feature_path}: {e}")
                problematic_files.append(feature_path)
    
    print("\nUnique Feature Shapes:")
    unique_shapes = set(feature_shapes)
    for shape in unique_shapes:
        print(shape)
    
    print(f"\nProblematic Files: {problematic_files}")

# Replace with your metadata path
metadata_path = r'E:\Project_2\metadata\reduced_audio_metadata.json'
diagnose_feature_extraction(metadata_path)