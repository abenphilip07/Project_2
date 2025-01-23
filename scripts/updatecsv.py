import pandas as pd

# Load the metadata CSV file
metadata_path = r"E:\Project_2\data\FakeAVCeleb_Sampled\metadata_updated.csv"
df = pd.read_csv(metadata_path)

# Update the 'full_path' column to combine 'full_path' and 'path'
df['full_path'] = df['full_path'].astype(str) + "\\" + df['path'].astype(str)

# Save the updated metadata back to CSV
updated_metadata_path = r"E:\Project_2\data\FakeAVCeleb_Sampled\metadata_updated1.csv"
df.to_csv(updated_metadata_path, index=False)

print("Metadata updated successfully! Saved to:", updated_metadata_path)
