import os

try:
    with open('/mnt/data1tb/vinh/TemporalGAN/corrupted_files.txt', 'r') as f:
        corrupt_paths = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print("corrupted_files.txt not found. Did the scanner find no corrupted files or did it fail?")
    exit(1)

bad_triplets = set()
for path in corrupt_paths:
    # Example path: /.../dataset/o2s_splited/train/triplet_vh/s1/12345.npy
    parts = path.split('/')
    filename = parts[-1]
    split_triplet_dir = '/'.join(parts[:-2])
    # split_triplet_dir is like /.../dataset/o2s_splited/train/triplet_vh
    bad_triplets.add(os.path.join(split_triplet_dir, filename))

removed_count = 0
triplet_count = 0
for base_path in bad_triplets:
    split_triplet_dir = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    print(f"Removing corrupted triplet: {filename} in {split_triplet_dir}")
    triplet_count += 1
    
    for mod in ['s1', 's2', 'lc']:
        file_to_remove = os.path.join(split_triplet_dir, mod, filename)
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)
            removed_count += 1

print(f"\nDone! Destroyed {triplet_count} corrupted triplets ({removed_count} individual files).")
