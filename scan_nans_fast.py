import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm

def check_file(path):
    try:
        arr = np.load(path)
        if np.isnan(arr).any() or np.isinf(arr).any():
            return path
    except Exception as e:
        return path  # Log as corrupted if it can't even load
    return None

if __name__ == '__main__':
    base_dir = '/mnt/data1tb/vinh/TemporalGAN/dataset/o2s_splited'
    all_files = []
    
    # Collect all .npy files in train/valid/test, triplet_vv/triplet_vh, s1/s2/lc
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.npy'):
                all_files.append(os.path.join(root, f))
                
    print(f"Total .npy files to scan: {len(all_files)}")
    
    corrupted = []
    with Pool(os.cpu_count()) as p:
        for res in tqdm(p.imap_unordered(check_file, all_files), total=len(all_files), desc="Scanning dataset"):
            if res is not None:
                corrupted.append(res)
                
    print(f"\nFound {len(corrupted)} corrupted files.")
    with open('/mnt/data1tb/vinh/TemporalGAN/corrupted_files.txt', 'w') as f:
        for c in corrupted:
            f.write(c + '\n')
    print("List saved to /mnt/data1tb/vinh/TemporalGAN/corrupted_files.txt")
