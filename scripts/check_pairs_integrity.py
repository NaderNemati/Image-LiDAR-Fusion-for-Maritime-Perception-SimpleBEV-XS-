import csv, pathlib, hashlib, collections, sys

csv_path = pathlib.Path("datasets/maritime_pairs.csv")
if not csv_path.exists():
    print(f"[ERROR] {csv_path} not found")
    sys.exit(1)

pairs = []
with csv_path.open(newline="") as f:
    rdr = csv.reader(f)
    header = next(rdr, None)  # skip header if present
    for row in rdr:
        # robust: skip malformed/short rows, only use first 2 columns
        if not row or len(row) < 2:
            continue
        img_path = row[0].strip()
        npy_path = row[2].strip() if len(row) >= 3 and row[2].strip().endswith(".npy") else row[1].strip()
        pairs.append((img_path, npy_path))

print(f"[INFO] loaded {len(pairs)} pairs from {csv_path}")

missing = 0
h2imgs = collections.defaultdict(list)

for img, npy in pairs:
    p = pathlib.Path(npy).expanduser()
    if not p.exists():
        missing += 1
        continue
    # hash file bytes to detect identical LiDAR contents reused across images
    h = hashlib.md5(p.read_bytes()).hexdigest()
    h2imgs[h].append((pathlib.Path(img).name, p.name))

if missing:
    print(f"[WARN] {missing} listed lidar .npy files are missing on disk.")

# Show top reuse groups
groups = sorted(h2imgs.items(), key=lambda kv: -len(kv[1]))
print(f"[INFO] unique lidar payloads: {len(groups)}")
for h, lst in groups[:10]:
    print(f"{len(lst):4d} frames share LiDAR hash {h}")
    for img, npy in lst[:6]:
        print("   ", f"{img:>24} -> {npy}")
    if len(lst) > 6:
        print("    ...")

