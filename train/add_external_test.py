from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def locate_pair(root: Path):
    candidates = [
        (root / "images", root / "labels"),
        (root / "test" / "images", root / "test" / "labels"),
        (root / "train" / "images", root / "train" / "labels"),
        (root / "valid" / "images", root / "valid" / "labels"),
    ]
    for images, labels in candidates:
        if images.is_dir() and labels.is_dir():
            return images, labels
    raise FileNotFoundError(
        "Could not find images/labels. Expected images+labels, "
        "test/images+test/labels, train/images+train/labels, or valid/images+valid/labels."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--dataset", default=Path("dataset"), type=Path)
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()

    images_src, labels_src = locate_pair(args.source.resolve())
    images_dst = args.dataset.resolve() / "images" / "test_real"
    labels_dst = args.dataset.resolve() / "labels" / "test_real"
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    if args.clear:
        for p in images_dst.iterdir():
            if p.is_file():
                p.unlink()
        for p in labels_dst.iterdir():
            if p.is_file():
                p.unlink()

    n = 0
    for image in sorted(images_src.iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMG_EXTS:
            continue
        label = labels_src / f"{image.stem}.txt"
        if not label.exists():
            raise FileNotFoundError(
                f"Missing label for {image.name}. Quantitative external testing requires labels."
            )

        # Tomorrow's filenames can include scenario prefixes such as far_, near_, occlusion_.
        dst_stem = f"external__{image.stem}"
        shutil.copy2(image, images_dst / f"{dst_stem}{image.suffix.lower()}")
        shutil.copy2(label, labels_dst / f"{dst_stem}.txt")
        n += 1

    print(f"Added {n} labeled external-test images.")
    print("Do NOT retrain/tune based on this test set before reporting the final score.")


if __name__ == "__main__":
    main()
