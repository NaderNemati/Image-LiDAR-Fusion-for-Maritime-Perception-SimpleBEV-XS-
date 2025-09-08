import sys, os
from pathlib import Path

def list_files(p: Path, max_show=5):
    files = [f for f in p.iterdir() if f.is_file()]
    files.sort()
    return files[:max_show], len(files)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_source.py <SRC_ROOT>")
        return
    root = Path(sys.argv[1]).expanduser().resolve()
    print("ROOT:", root)
    if not root.exists():
        print("Not found.")
        return

    image_dirs = []
    lidar_dirs = []
    for d in root.rglob("*"):
        if not d.is_dir(): continue
        # case-insensitive match by suffix
        imgs = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in {".jpg",".jpeg",".png"}]
        lids = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in {".pcd",".bin"}]
        if len(imgs) >= 5:
            image_dirs.append((d, len(imgs)))
        if len(lids) >= 5:
            lidar_dirs.append((d, len(lids)))

    print("\nCandidate image dirs (>=5 files):")
    for d,c in sorted(image_dirs, key=lambda x: -x[1])[:10]:
        ex, _ = list_files(d)
        print(f"  {d}  (#={c})  e.g.: {[e.name for e in ex]}")

    print("\nCandidate LiDAR dirs (>=5 files):")
    for d,c in sorted(lidar_dirs, key=lambda x: -x[1])[:10]:
        ex, _ = list_files(d)
        print(f"  {d}  (#={c})  e.g.: {[e.name for e in ex]}")

if __name__ == "__main__":
    main()
