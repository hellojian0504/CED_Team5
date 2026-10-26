from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

CLASS_NAMES = ["Scylla", "TrojanHorse", "cube_10cm", "octahedron_10cm"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=Path("dataset/data.yaml"))
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--op-conf", type=float, default=0.25,
                    help="Operational confidence used only for negative-frame FP reporting.")
    args = ap.parse_args()

    model = YOLO(str(args.weights))

    # Keep validation confidence low so PR/mAP curves are not artificially truncated.
    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        verbose=False,
    )

    print("\n=== Overall metrics ===")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP75    : {metrics.box.map75:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")

    maps = list(metrics.box.maps)
    print("\n=== Per-class mAP50-95 ===")
    for i, value in enumerate(maps):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
        print(f"{i} {name:16s}: {float(value):.4f}")

    # Extra deployment-style check: false positives on test frames with empty labels.
    if args.split == "test":
        dataset_root = args.data.resolve().parent
        img_dir = dataset_root / "images" / "test_real"
        lbl_dir = dataset_root / "labels" / "test_real"
        negatives = []
        if img_dir.is_dir():
            for image in sorted(img_dir.iterdir()):
                if image.suffix.lower() not in IMG_EXTS:
                    continue
                label = lbl_dir / f"{image.stem}.txt"
                if label.exists() and not label.read_text(encoding="utf-8").strip():
                    negatives.append(str(image))

        if negatives:
            preds = model.predict(
                source=negatives,
                imgsz=args.imgsz,
                conf=args.op_conf,
                device=args.device,
                verbose=False,
            )
            total_fp = sum(len(r.boxes) for r in preds)
            frames_with_fp = sum(1 for r in preds if len(r.boxes) > 0)
            print("\n=== Negative-frame false positives ===")
            print(f"negative frames            : {len(negatives)}")
            print(f"frames with >=1 detection  : {frames_with_fp}")
            print(f"total false detections      : {total_fp}")
            print(f"FP per negative frame       : {total_fp / len(negatives):.4f}")
            print(f"operational conf threshold  : {args.op_conf:.3f}")
        else:
            print("\nNo empty-label negative frames found in test_real.")


if __name__ == "__main__":
    main()
