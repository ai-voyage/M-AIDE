'''Merge batch activations extraction into one per layer'''

import os
import numpy as np
import glob

batch_dir = "/Path/to/mixtral_residuals_all_layers"
merged_dir = os.path.join(batch_dir, "merged")
os.makedirs(merged_dir, exist_ok=True)

for layer_id in range(32):
    pattern = os.path.join(batch_dir, f"layer_{layer_id}_batch_*.npy")
    batch_files = sorted(glob.glob(pattern))

    if not batch_files:
        print(f" No batches found for layer {layer_id}")
        continue

    print(f" Merging layer {layer_id} ({len(batch_files)} batches)...")
    arrays = [np.load(f) for f in batch_files]
    merged = np.concatenate(arrays, axis=0)

    out_path = os.path.join(merged_dir, f"layer_{layer_id}.npy")
    np.save(out_path, merged)
    print(f" Merged and saved {out_path}: {merged.shape}")