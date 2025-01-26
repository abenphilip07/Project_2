import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import json
import numpy as np
from models.FCMT import FCMT
import yaml
import os

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load metadata
metadata_path = "E:\\Project_2\\metadata\\resnet_metadata1.json"
with open(metadata_path, "r") as f:
    metadata = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize FCMT model
model = FCMT(
    input_dim=config['input_dim'],
    d_model=config['d_model'],
    n_heads=config['n_heads'],
    num_layers=config['num_layers'],
    d_ff=config['d_ff'],
    dropout=config['dropout']
).to(device)

model.eval()  # Set model to evaluation mode

# Directory to store transformed features
output_dir = "E:\\Project_2\\transformed_features"
os.makedirs(output_dir, exist_ok=True)

# New metadata dictionary
new_metadata = {}

for video_id, video_data in metadata.items():
    new_metadata[video_id] = {
        "original_video": video_data["original_video"],
        "total_chunks": video_data["total_chunks"],
        "chunks": []
    }

    for chunk in video_data["chunks"]:
        # Load chunk features
        features = np.load(chunk["feature_path"])
        feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        # Get transformed feature sequence
        with torch.no_grad():
            transformed_features = model(feature_tensor)
        
        # Save transformed features to new path
        transformed_feature_path = os.path.join(output_dir, f"{video_id}_chunk{chunk['chunk_index']}_transformed.npy")
        np.save(transformed_feature_path, transformed_features.cpu().numpy())

        # Update new metadata
        new_metadata[video_id]["chunks"].append({
            "chunk_index": chunk["chunk_index"],
            "audio_path": chunk["audio_path"],
            "video_path": chunk["video_path"],
            "original_feature_path": chunk["feature_path"],
            "transformed_feature_path": transformed_feature_path
        })

        print(f"Processed chunk {chunk['chunk_index']} for video: {video_id}")

# Save updated metadata with transformed feature paths
new_metadata_path = "E:\\Project_2\\metadata\\transformed_metadata.json"
with open(new_metadata_path, "w") as f:
    json.dump(new_metadata, f, indent=4)

print(f"Updated metadata saved at: {new_metadata_path}")
