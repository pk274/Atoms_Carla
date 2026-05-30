import numpy as np
from pathlib import Path

frames_dir = Path("data/TFV6/baseline_data/frames")
npz_files = sorted(frames_dir.glob("run_*.npz"))

# Aggregate class counts across all files
total_counts = {}
total_pixels = 0
sample_file = npz_files[0]

d = np.load(sample_file, allow_pickle=True)
print("NPZ keys:", list(d.keys()))
seg = d["seg_red_wide"]
print(f"seg_red_wide shape: {seg.shape}, dtype: {seg.dtype}, min: {seg.min()}, max: {seg.max()}")

for npz_path in npz_files:
    d = np.load(npz_path, allow_pickle=True)
    seg = d["seg_red_wide"]
    vals, counts = np.unique(seg, return_counts=True)
    for v, c in zip(vals.tolist(), counts.tolist()):
        total_counts[v] = total_counts.get(v, 0) + c
    total_pixels += seg.size

print(f"\nAggregated over {len(npz_files)} files, {total_pixels} total pixels:")
print("Class ID | Count | Fraction")
for v, c in sorted(total_counts.items(), key=lambda x: -x[1]):
    print(f"  {v:5d}  | {c:10d} | {100*c/total_pixels:.2f}%")
