import os
import shutil
import pandas as pd

# Define paths
original_metadata_path = r"E:\FakeAVCeleb_v1.2\meta_data.csv"
new_data_folder = r"E:\Project_2\data\FakeAVCeleb_Sampled"
new_metadata_path = r"E:\Project_2\data\FakeAVCeleb_Sampled\metadata_updated.csv"

# Ensure new folder exists
os.makedirs(new_data_folder, exist_ok=True)

# Load metadata
df = pd.read_csv(original_metadata_path)

# Select 100 videos per label, randomly choosing race and gender
selected_videos = pd.DataFrame()

for label in ['FakeVideo-FakeAudio', 'FakeVideo-RealAudio', 'RealVideo-FakeAudio', 'RealVideo-RealAudio']:
    label_videos = df[df['type'] == label]
    
    # Randomly sample 100 videos irrespective of race and gender
    if len(label_videos) >= 100:
        selected = label_videos.sample(n=100, random_state=42)
    else:
        print(f"Warning: Not enough videos for {label}, selecting available {len(label_videos)} videos.")
        selected = label_videos

    selected_videos = pd.concat([selected_videos, selected], ignore_index=True)

# Copy selected videos to new folder without subfolders
new_paths = []
valid_videos = pd.DataFrame()

for index, row in selected_videos.iterrows():
    base_path = "E:/"
    old_path = os.path.normpath(os.path.join(base_path, row['full_path'], row['path']))
    new_filename = f"{row['source']}_{row['type']}_{row['race']}_{row['gender']}.mp4"
    new_path = os.path.join(new_data_folder, new_filename)
    
    # Check if file exists
    if os.path.exists(old_path):
        try:
            shutil.copy2(old_path, new_path)
            new_paths.append(new_filename)
            valid_videos = pd.concat([valid_videos, row.to_frame().T], ignore_index=True)
        except Exception as e:
            print(f"Error copying {old_path}: {e}")
    else:
        print(f"Warning: File {old_path} not found. Skipping.")

# Update only valid entries
valid_videos['path'] = new_paths
valid_videos['full_path'] = new_data_folder  # Update full path to the new folder location

# Save new metadata CSV
valid_videos.to_csv(new_metadata_path, index=False)

print("Data sampling completed successfully!")
