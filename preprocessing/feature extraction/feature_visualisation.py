import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

def visualize_fake_real_features_1000(metadata_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    features = []
    labels = []
    
    for video_name, video_data in metadata.items():
        is_fake = 'FakeVideo' in video_name
        for chunk in video_data['chunks']:
            feature_path = chunk['feature_path']
            feature = np.load(feature_path)
            
            if feature.shape == (1000,):
                features.append(feature)
                labels.append('Fake' if is_fake else 'Real')
    
    features = np.array(features)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # PCA Visualization
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                          c=encoded_labels, 
                          cmap='viridis')
    plt.colorbar(scatter, label='Video Type')
    plt.title('PCA of 1000D Video Features: Fake vs Real')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_1000d_features.png'))
    plt.close()
    
    # Feature Distribution
    plt.figure(figsize=(15, 6))
    for label in set(labels):
        subset = features[np.array(labels) == label]
        sns.histplot(subset.flatten(), kde=True, label=label)
    plt.title('1000D Feature Distribution: Fake vs Real')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distribution_1000d.png'))
    plt.close()
    
    # Print summary
    print(f"Total 1000D features: {len(features)}")
    print(f"Fake videos: {labels.count('Fake')}")
    print(f"Real videos: {labels.count('Real')}")

# Usage
metadata_path = r'E:\Project_2\updated_metadata.json'
output_dir = "1000d_feature_visualizations"
visualize_fake_real_features_1000(metadata_path, output_dir)