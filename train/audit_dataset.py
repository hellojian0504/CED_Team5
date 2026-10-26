from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0:"Scylla", 1:"TrojanHorse", 2:"cube_10cm", 3:"octahedron_10cm"}


def sha256(path: Path, chunk=1024*1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def audit_split(root: Path, split: str):
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
    objects = Counter()
    empty = 0
    errors = []

    for image in images:
        label = lbl_dir / f"{image.stem}.txt"
        if not label.exists():
            errors.append(f"missing label: {image.name}")
            continue
        text = label.read_text(encoding="utf-8").strip()
        if not text:
            empty += 1
            continue
        for ln, line in enumerate(text.splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{label.name}:{ln}: expected 5 fields, got {len(parts)}")
                continue
            try:
                cls = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
            except Exception:
                errors.append(f"{label.name}:{ln}: non-numeric label")
                continue
            if cls not in CLASS_NAMES:
                errors.append(f"{label.name}:{ln}: invalid class {cls}")
            if any(v < 0.0 or v > 1.0 for v in vals):
                errors.append(f"{label.name}:{ln}: normalized bbox value outside [0,1]: {vals}")
            if vals[2] <= 0 or vals[3] <= 0:
                errors.append(f"{label.name}:{ln}: non-positive width/height: {vals[2:]}")
            objects[cls] += 1

    return images, objects, empty, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("dataset"))
    args = ap.parse_args()
    root = args.dataset.resolve()

    splits = ["train_synth", "train_real", "val_real", "val_synth", "test_real"]
    all_errors = []
    hash_locations = defaultdict(list)

    for split in splits:
        images, objects, empty, errors = audit_split(root, split)
        all_errors.extend(f"[{split}] {e}" for e in errors)
        print(f"\n[{split}] images={len(images)} empty_labels={empty}")
        for cls in range(4):
            print(f"  {cls} {CLASS_NAMES[cls]:16s}: {objects[cls]} objects")
        for image in images:
            hash_locations[sha256(image)].append((split, image.name))

    leakage = []
    for h, locs in hash_locations.items():
        split_names = {x[0] for x in locs}
        if len(split_names) > 1:
            leakage.append(locs)

    print("\n=== Cross-split exact duplicate check ===")
    if leakage:
        for locs in leakage[:20]:
            print("DUPLICATE:", locs)
        all_errors.append(f"{len(leakage)} exact image hashes appear across multiple splits")
    else:
        print("No exact image duplicates across splits.")

    if all_errors:
        print("\n=== ERRORS ===")
        for e in all_errors[:100]:
            print(e)
        raise SystemExit(f"Audit failed with {len(all_errors)} issue(s).")
    print("\nAudit passed.")


if __name__ == "__main__":
    main()
