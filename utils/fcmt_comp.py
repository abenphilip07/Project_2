import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_metadata(metadata_path):
    """Load metadata from JSON file"""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def compare_features(original_path, transformed_path):
    """Compare original and transformed feature files"""
    try:
        original_features = np.load(original_path)
        transformed_features = np.load(transformed_path)
        
        # Flatten features for easier comparison
        orig_flat = original_features.flatten()
        trans_flat = transformed_features.flatten()
        
        # Detailed difference calculations
        abs_diff = np.abs(orig_flat - trans_flat)
        
        return {
            'mean_difference': np.mean(abs_diff),
            'max_difference': np.max(abs_diff),
            'min_difference': np.min(abs_diff),
            'std_difference': np.std(abs_diff),
            'median_difference': np.median(abs_diff),
            'original_shape': original_features.shape,
            'transformed_shape': transformed_features.shape
        }
    except Exception as e:
        print(f"Error comparing features: {e}")
        return None

def print_detailed_differences(metadata):
    """Print detailed differences to terminal"""
    print("\n--- Feature Transformation Differences ---")
    print("-" * 50)
    
    for video_name, video_info in metadata.items():
        print(f"\nVideo: {video_name}")
        print("-" * 30)
        
        for chunk_index, chunk in enumerate(video_info['chunks']):
            result = compare_features(
                chunk['original_feature_path'], 
                chunk['reduced_feature_path']
            )
            
            if result:
                print(f"Chunk {chunk_index}:")
                print(f"  Original Shape: {result['original_shape']}")
                print(f"  Transformed Shape: {result['transformed_shape']}")
                print(f"  Mean Difference: {result['mean_difference']:.4f}")
                print(f"  Max Difference: {result['max_difference']:.4f}")
                print(f"  Min Difference: {result['min_difference']:.4f}")
                print(f"  Std Deviation of Differences: {result['std_difference']:.4f}")
                print(f"  Median Difference: {result['median_difference']:.4f}")

# Main execution
metadata_path = r'E:\Project_2\metadata\reduced_metadata1.json'
metadata = load_metadata(metadata_path)
print_detailed_differences(metadata)