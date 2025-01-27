import numpy as np
import matplotlib.pyplot as plt
import json

def compare_features(original_path, reduced_path):
    """Compare original and reduced feature files"""
    original_features = np.load(original_path)
    reduced_features = np.load(reduced_path)
    
    plt.figure(figsize=(15, 5))
    
    # Handle 1D or 2D arrays
    plt.subplot(1, 2, 1)
    plt.title('Original Features')
    if original_features.ndim == 1:
        plt.plot(original_features)
    else:
        plt.imshow(original_features, aspect='auto', cmap='viridis')
        plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title('Reduced Features')
    if reduced_features.ndim == 1:
        plt.plot(reduced_features)
    else:
        plt.imshow(reduced_features, aspect='auto', cmap='viridis')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('feature_comparison.png')
    plt.close()
    
    print("Original Features Shape:", original_features.shape)
    print("Reduced Features Shape:", reduced_features.shape)

# Load metadata
metadata_path = r'E:\Project_2\metadata\reduced_audio_metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Compare features for each video chunk
for video_name, video_info in metadata.items():
    for chunk in video_info['chunks']:
        compare_features(
            chunk['original_feature_path'], 
            chunk['reduced_audio_feature_path']
        )