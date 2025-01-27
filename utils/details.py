import os
import pandas as pd

# Define paths
new_metadata_path = r"E:\Project_2\data\FakeAVCeleb_Sampled\metadata_updated.csv"
new_data_folder = r"E:\Project_2\data\FakeAVCeleb_Sampled"

# Load updated metadata
df = pd.read_csv(new_metadata_path)

# Generate summary statistics
def generate_report(df, new_data_folder):
    total_videos = len(df)
    label_counts = df['type'].value_counts()
    race_gender_counts = df.groupby(['race', 'gender']).size()

    # Count number of files physically in the new folder
    actual_files = len([f for f in os.listdir(new_data_folder) if f.endswith(".mp4")])

    print("===== FakeAVCeleb Sampled Dataset Report =====")
    print(f"Total videos (according to metadata): {total_videos}")
    print(f"Total videos (actual in folder): {actual_files}")
    print("\nVideos per label:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} videos")
    
    print("\nVideos per race and gender:")
    for (race, gender), count in race_gender_counts.items():
        print(f"  {race} - {gender}: {count} videos")

    # Find missing files
    missing_files = [row['path'] for _, row in df.iterrows() if not os.path.exists(os.path.join(new_data_folder, row['path']))]

    if missing_files:
        print(f"\nTotal Missing Files: {len(missing_files)}")
        print("The following files are missing:")
        for missing in missing_files:
            print(f"  {missing}")
    else:
        print("\nAll files are present.")

    # Check for duplicate file paths
    duplicate_entries = df[df.duplicated(subset=['path'], keep=False)]

    if not duplicate_entries.empty:
        print(f"\nWarning! Duplicate file paths detected in metadata: {len(duplicate_entries)}")
        print(duplicate_entries[['path']].drop_duplicates())
    else:
        print("\nNo duplicate file paths in metadata.")



# Run the report generation function
generate_report(df, new_data_folder)
