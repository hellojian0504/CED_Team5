from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=Path("dataset/data_real_finetune.yaml"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--name", default="real_finetune")
    ap.add_argument("--project", default="runs/detect")
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        patience=12,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        project=args.project,
        name=args.name,
        seed=42,
        deterministic=True,

        # Lower-LR real-domain adaptation.
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.05,

        # Moderate augmentation: keep the real appearance recognizable.
        hsv_h=0.010,
        hsv_s=0.30,
        hsv_v=0.30,
        degrees=5.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.50,
        flipud=0.0,
        perspective=0.0003,
        mosaic=0.50,
        close_mosaic=10,
        mixup=0.0,
        cutmix=0.0,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print("Fine-tuned best.pt:", best)
    print("Compare this against the mixed-training best.pt on val_real and the future external test.")


if __name__ == "__main__":
    main()
