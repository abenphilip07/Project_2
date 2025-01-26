import os
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import cv2

# Initialize the pretrained ResNet-50 model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()  # Move model to device and set to evaluation mode

# Define transformations for frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Feature extraction function
def extract_features_from_video(video_path, model, transform, frame_sample_rate=10):
    """Extract temporal features from a video file using a pretrained model.
    Args:
        video_path (str): Path to the video file.
        model (torch.nn.Module): Pretrained model for feature extraction.
        transform (torchvision.transforms.Compose): Transformations for input frames.
        frame_sample_rate (int): Sample every Nth frame from the video.

    Returns:
        np.ndarray: Aggregated feature vector for the video.
    """
    cap = cv2.VideoCapture(video_path)
    frame_features = []
    frame_count = 0

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample every Nth frame
            if frame_count % frame_sample_rate == 0:
                # Apply transformations and move to the appropriate device
                frame_tensor = transform(frame).unsqueeze(0)  # Add batch dimension
                frame_tensor = frame_tensor.to(device)

                # Extract features
                features = model(frame_tensor)
                frame_features.append(features.squeeze().cpu().numpy())

            frame_count += 1

    cap.release()
    # Aggregate frame features (e.g., by averaging)
    if frame_features:
        return np.mean(frame_features, axis=0)
    else:
        return np.zeros(1000,)

# Load metadata
metadata_path =  r'E:\Project_2\metadata\chunks_metadata.json'  # Path to the metadata file
output_feature_dir = r'E:\Project_2\data\Features1'  # Directory to store extracted features
updated_metadata_path = r'E:\Project_2\metadata\resnet_metadata1.json' # Path to save updated metadata

os.makedirs(output_feature_dir, exist_ok=True)

# Read metadata
with open(metadata_path, "r") as f:
    metadata = json.load(f)

updated_metadata = {}

for video_name, video_data in metadata.items():
    updated_metadata[video_name] = {
        "original_video": video_data["original_video"],
        "total_chunks": video_data["total_chunks"],
        "chunks": []
    }

    for chunk in video_data["chunks"]:
        chunk_index = chunk["chunk_index"]
        video_path = chunk["video_path"]

        # Extract features using the pretrained model
        features = extract_features_from_video(video_path, model, transform)

        # Save features to a .npy file
        feature_file_name = f"{video_name}_chunk{chunk_index:03d}_features.npy"
        feature_file_path = os.path.join(output_feature_dir, feature_file_name)
        np.save(feature_file_path, features)

        # Update metadata
        updated_metadata[video_name]["chunks"].append({
            "chunk_index": chunk_index,
            "audio_path": chunk["audio_path"],
            "video_path": video_path,
            "feature_path": feature_file_path
        })

# Save updated metadata
with open(updated_metadata_path, "w") as f:
    json.dump(updated_metadata, f, indent=4)

print(f"Feature extraction completed. Updated metadata saved to {updated_metadata_path}.")
