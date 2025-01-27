import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load metadata
metadata_path = r'E:\Project_2\metadata\transformed_audiometadata.json'
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Directory to store 1D reduced features
reduced_feature_dir = r'E:\Project_2\data\audio features\reduced_features'
os.makedirs(reduced_feature_dir, exist_ok=True)

# New metadata dictionary
new_metadata = {}

# Process each video in the metadata
for video_id, video_data in metadata.items():
    new_metadata[video_id] = {
        "original_video": video_data["original_video"],
        "total_chunks": video_data["total_chunks"],
        "chunks": []
    }

    for chunk in video_data["chunks"]:
        # Load the transformed features
        transformed_feature_path = chunk["transformed_audio_feature_path"]
        transformed_features = np.load(transformed_feature_path)  # Shape: (1, 512, 1000)

        # Apply max pooling along the sequence dimension (dim=1)
        transformed_tensor = torch.tensor(transformed_features, dtype=torch.float32)  # Convert to tensor
        pooled_tensor = F.adaptive_max_pool1d(transformed_tensor.squeeze(0).permute(1, 0), 1)  # Shape: (1000, 1)
        reduced_features = pooled_tensor.squeeze(1).numpy()  # Final shape: (1000,)

        # Save the reduced features
        reduced_feature_path = os.path.join(reduced_feature_dir, f"{video_id}_chunk{chunk['chunk_index']}_reduced.npy")
        np.save(reduced_feature_path, reduced_features)

        # Update new metadata
        new_metadata[video_id]["chunks"].append({
            "chunk_index": chunk["chunk_index"],
            "audio_path": chunk["audio_path"],
            "video_path": chunk["video_path"],
            "original_feature_path": chunk["original_feature_path"],
            "transformed_audio_feature_path": chunk["transformed_audio_feature_path"],
            "reduced_audio_feature_path": reduced_feature_path  # Add new path for reduced features
        })

        print(f"Processed chunk {chunk['chunk_index']} for video: {video_id}")

# Save updated metadata with reduced feature paths
new_metadata_path = "E:\\Project_2\\metadata\\reduced_audio_metadata.json"
with open(new_metadata_path, "w") as f:
    json.dump(new_metadata, f, indent=4)

print(f"Updated metadata saved at: {new_metadata_path}")
