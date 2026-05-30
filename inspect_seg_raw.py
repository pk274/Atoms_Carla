import cv2
import numpy as np
from pathlib import Path

# Raw LEAD semantics PNGs
seg_paths = [
    Path(r"D:\Carla_tfv6_data\data\carla_leaderboard2\data\noScenarios_baseline\Town03_Rep0_route_000403_route0_01_10_06_30_33\semantics\0000.png"),
    Path(r"D:\Carla_tfv6_data\data\carla_leaderboard2\data\noScenarios_baseline\Town03_Rep0_route_000403_route0_01_10_06_30_33\semantics\0050.png"),
    Path(r"D:\Carla_tfv6_data\data\carla_leaderboard2\data\noScenarios_baseline\Town03_Rep0_route_000403_route0_01_10_06_30_33\semantics\0100.png"),
]

for sf in seg_paths:
    if not sf.exists():
        print(f"NOT FOUND: {sf}")
        continue
    img = cv2.imread(str(sf), cv2.IMREAD_UNCHANGED)
    print(f"\n{sf.name}: shape={img.shape}, dtype={img.dtype}")
    if img.ndim == 3:
        for ci, cname in enumerate(["B(ch0)", "G(ch1)", "R(ch2)"]):
            ch = img[:, :, ci]
            vals, counts = np.unique(ch, return_counts=True)
            total = ch.size
            top = sorted(zip(vals.tolist(), counts.tolist()), key=lambda x: -x[1])[:15]
            top_str = ", ".join(f"{v}({100*c/total:.1f}%)" for v, c in top)
            print(f"  {cname}: [{top_str}]")
    else:
        vals, counts = np.unique(img, return_counts=True)
        total = img.size
        top = sorted(zip(vals.tolist(), counts.tolist()), key=lambda x: -x[1])[:15]
        print(f"  values: {top}")

# Also check a meta file to see if there's any class info
import lzma, pickle
meta_path = Path(r"D:\Carla_tfv6_data\data\carla_leaderboard2\data\noScenarios_baseline\Town03_Rep0_route_000403_route0_01_10_06_30_33\metas\0000.pkl")
if meta_path.exists():
    with open(meta_path, "rb") as f:
        meta = pickle.loads(lzma.decompress(f.read()))
    print("\nMeta keys:", list(meta.keys()))
    for k in sorted(meta.keys()):
        v = meta[k]
        if not isinstance(v, (list, dict, np.ndarray)) or (hasattr(v, '__len__') and len(v) < 10):
            print(f"  {k}: {v}")
