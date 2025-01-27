import os
import csv
import cv2

def chunk_videos(metadata_file, output_base_dir, chunk_size=30, overlap=5):
    """
    Chunk videos into segments with overlap and generate new metadata
    """
    os.makedirs(output_base_dir, exist_ok=True)
    chunks_dir = os.path.join(output_base_dir, 'chunks')
    os.makedirs(chunks_dir, exist_ok=True)
    
    output_metadata_path = os.path.join(output_base_dir, 'chunks_metadata.csv')
    output_metadata = []
    
    with open(metadata_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            video_path = row['full_path']
            source_filename = row['filename']
            
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            for i in range(0, len(frames) - chunk_size + 1, chunk_size - overlap):
                chunk = frames[i:i + chunk_size]
                
                chunk_filename = f'video_{i//chunk_size:04d}.mp4'
                chunk_path = os.path.join(chunks_dir, chunk_filename)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(chunk_path, fourcc, fps, 
                                      (chunk[0].shape[1], chunk[0].shape[0]))
                for frame in chunk:
                    out.write(frame)
                out.release()
                
                chunk_metadata = {
                    'source': row['source'],
                    'target1': row['target1'],
                    'target2': row['target2'],
                    'method': row['method'],
                    'category': row['category'],
                    'type': row['type'],
                    'race': row['race'],
                    'gender': row['gender'],
                    'filename': chunk_filename,
                    'chunk_index': i//chunk_size,
                    'original_source_filename': source_filename
                }
                output_metadata.append(chunk_metadata)
    
    if output_metadata:
        keys = list(output_metadata[0].keys())
        with open(output_metadata_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(output_metadata)
        
        print(f"Created {len(output_metadata)} video chunks with metadata at {output_metadata_path}")

# Example usage
metadata_file = r'E:\Project_2\data\FakeAVCeleb_Sampled\chunks\chunks_metadata.csv'
output_base_dir = r'E:\Project_2\data\FakeAVCeleb_Sampled\chunks'
chunk_videos(metadata_file, output_base_dir)