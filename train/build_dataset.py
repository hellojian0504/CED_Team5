from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = {
    0: "Scylla",
    1: "TrojanHorse",
    2: "cube_10cm",
    3: "octahedron_10cm",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_real_pair(root: Path) -> tuple[Path, Path]:
    candidates = [
        (root / "train" / "images", root / "train" / "labels"),
        (root / "images", root / "labels"),
    ]
    for images, labels in candidates:
        if images.is_dir() and labels.is_dir():
            return images, labels
    raise FileNotFoundError(
        f"Could not find real dataset image/label pair under {root}. "
        "Expected train/images + train/labels or images + labels."
    )


def iter_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def label_path(labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def read_class_ids(label_file: Path) -> list[int]:
    ids = []
    text = label_file.read_text(encoding="utf-8").strip()
    if not text:
        return ids
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed YOLO label: {label_file}: {line!r}")
        ids.append(int(float(parts[0])))
    return ids


def validate_pair(image: Path, label: Path):
    if not label.exists():
        raise FileNotFoundError(f"Missing label for {image}: {label}")
    for cls in read_class_ids(label):
        if cls not in CLASS_NAMES:
            raise ValueError(f"Unexpected class id {cls} in {label}. Expected only 0..3.")


def copy_pair(image: Path, label: Path, dst_img: Path, dst_lbl: Path, prefix: str):
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    new_stem = f"{prefix}__{image.stem}"
    image_dst = dst_img / f"{new_stem}{image.suffix.lower()}"
    label_dst = dst_lbl / f"{new_stem}.txt"
    shutil.copy2(image, image_dst)
    shutil.copy2(label, label_dst)
    return image_dst, label_dst


def infer_class_from_filename(name: str) -> str:
    n = name.lower()
    if "scylla" in n:
        return "scylla"
    if "horse" in n:
        return "horse"
    if "cube" in n:
        return "cube"
    if "octa" in n:
        return "octa"
    return "unknown"


def block_group(image: Path, block_size: int) -> str:
    """
    Nearby sequential captures are grouped together to reduce train/val leakage.
    e.g. cube001..cube010 -> same block.
    """
    stem = image.stem.lower()
    m = re.search(r"(scylla|horse|cube|octa)[^0-9]*([0-9]+)", stem)
    if m:
        cls, num = m.group(1), int(m.group(2))
        block = (num - 1) // block_size
        return f"{cls}_block_{block:03d}"
    # Fallback: each image is its own group.
    return stem


def split_real_groups(images: list[Path], val_fraction: float, seed: int, block_size: int):
    groups = defaultdict(list)
    for image in images:
        groups[block_group(image, block_size)].append(image)

    # Split independently per inferred class so val stays balanced.
    by_class = defaultdict(list)
    for g in groups:
        cls = g.split("_block_")[0] if "_block_" in g else infer_class_from_filename(g)
        by_class[cls].append(g)

    rng = random.Random(seed)
    val_groups = set()
    for cls, gs in sorted(by_class.items()):
        gs = sorted(gs)
        rng.shuffle(gs)
        n_val = max(1, round(len(gs) * val_fraction)) if len(gs) > 1 else 0
        val_groups.update(gs[:n_val])

    train, val = [], []
    for group_name, ims in groups.items():
        (val if group_name in val_groups else train).extend(ims)
    return sorted(train), sorted(val)


def write_yaml(path: Path, dataset_root: Path, train, val, test=None):
    def yaml_value(v):
        if isinstance(v, list):
            return "\n" + "\n".join(f"  - {x}" for x in v)
        return str(v)

    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        f"train: {yaml_value(train)}",
        f"val: {yaml_value(val)}",
    ]
    if test is not None:
        lines.append(f"test: {yaml_value(test)}")
    lines += [
        "",
        "nc: 4",
        "names:",
        "  0: Scylla",
        "  1: TrojanHorse",
        "  2: cube_10cm",
        "  3: octahedron_10cm",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True, type=Path)
    ap.add_argument("--real-cube-octa", required=True, type=Path)
    ap.add_argument("--real-horse-scylla", required=True, type=Path)
    ap.add_argument("--out", default=Path("dataset"), type=Path)
    ap.add_argument("--real-val-fraction", type=float, default=0.20)
    ap.add_argument("--block-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)

    split_dirs = ["train_synth", "train_real", "val_real", "val_synth", "test_real"]
    for split in split_dirs:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    def add(image: Path, label: Path, split: str, source: str):
        validate_pair(image, label)
        dst_img, _ = copy_pair(
            image, label,
            out / "images" / split,
            out / "labels" / split,
            prefix=source,
        )
        cls_ids = read_class_ids(label)
        manifest_rows.append({
            "split": split,
            "source": source,
            "source_image": str(image),
            "output_image": str(dst_img),
            "class_ids": ",".join(map(str, cls_ids)),
        })

    # Synthetic: preserve the generator's existing train/val split.
    for split_in, split_out in [("train", "train_synth"), ("val", "val_synth")]:
        images_dir = args.synth / "images" / split_in
        labels_dir = args.synth / "labels" / split_in
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing synthetic split: {images_dir} / {labels_dir}")
        for image in iter_images(images_dir):
            add(image, label_path(labels_dir, image), split_out, f"synth_{split_in}")

    # Real datasets: split by sequential blocks, not individual random images.
    for source_name, real_root in [
        ("real_cube_octa", args.real_cube_octa),
        ("real_horse_scylla", args.real_horse_scylla),
    ]:
        images_dir, labels_dir = resolve_real_pair(real_root)
        images = iter_images(images_dir)
        train_images, val_images = split_real_groups(
            images,
            val_fraction=args.real_val_fraction,
            seed=args.seed,
            block_size=args.block_size,
        )
        for image in train_images:
            add(image, label_path(labels_dir, image), "train_real", source_name)
        for image in val_images:
            add(image, label_path(labels_dir, image), "val_real", source_name)

    # Main YAML: checkpoint selection is based on REAL validation.
    write_yaml(
        out / "data.yaml",
        out,
        train=["images/train_synth", "images/train_real"],
        val="images/val_real",
        test="images/test_real",
    )

    # Optional YAMLs for diagnostics / second-stage fine tuning.
    write_yaml(
        out / "data_real_finetune.yaml",
        out,
        train="images/train_real",
        val="images/val_real",
        test="images/test_real",
    )
    write_yaml(
        out / "data_synth_diag.yaml",
        out,
        train="images/train_synth",
        val="images/val_synth",
    )
    write_yaml(
        out / "data_mixed_val.yaml",
        out,
        train=["images/train_synth", "images/train_real"],
        val=["images/val_real", "images/val_synth"],
        test="images/test_real",
    )

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "source", "source_image", "output_image", "class_ids"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Built dataset at: {out}")
    counts = defaultdict(int)
    for row in manifest_rows:
        counts[row["split"]] += 1
    for k in split_dirs:
        print(f"{k:12s}: {counts[k]:4d} images")
    print("test_real stays empty until a NEW labeled real capture is added.")


if __name__ == "__main__":
    main()
