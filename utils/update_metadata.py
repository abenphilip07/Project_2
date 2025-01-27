import json

# Function to update the metadata
def update_metadata(input_file, output_file):
    with open(input_file, 'r') as f:
        metadata = json.load(f)

    updated_metadata = {}

    for video_id, details in metadata.items():
        updated_chunks = []
        for chunk in details['chunks']:
            # Update feature_path to video_feature_path with the new directory
            chunk['video_feature_path'] = chunk.pop('feature_path').replace(
                "E:\\Project_2\\data\\Features1", "E:\\Project_2\\data\\video features\\Features1"
            )
            updated_chunks.append(chunk)

        # Construct the updated metadata
        updated_metadata[video_id] = {
            "original_video": details['original_video'],
            "total_chunks": details['total_chunks'],
            "chunks": updated_chunks
        }

    # Write the updated metadata to the output file
    with open(output_file, 'w') as f:
        json.dump(updated_metadata, f, indent=4)

# Example usage
input_file = r'E:\Project_2\metadata\resnet_metadata1.json'  # Replace with your input JSON file
output_file = r'E:\Project_2\metadata\resnet_metadata2.json'  # Replace with your desired output file
update_metadata(input_file, output_file)