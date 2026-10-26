from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def extract(metrics):
    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset/data.yaml"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--include-m", action="store_true")
    ap.add_argument("--project", default="runs/detect")
    args = ap.parse_args()

    dataset_root = args.data.resolve().parent
    synth_diag = dataset_root / "data_synth_diag.yaml"

    experiments = [
        ("y11n_640_baseline", "yolo11n.pt", 640),
        ("y11n_960", "yolo11n.pt", 960),
        ("y11s_960", "yolo11s.pt", 960),
    ]
    if args.include_m:
        experiments.append(("y11m_960", "yolo11m.pt", 960))

    rows = []

    for name, model_name, imgsz in experiments:
        print("\n" + "=" * 80)
        print(f"TRAINING {name}: {model_name}, imgsz={imgsz}")
        print("=" * 80)

        model = YOLO(model_name)
        model.train(
            data=str(args.data.resolve()),
            epochs=args.epochs,
            patience=40,
            imgsz=imgsz,
            batch=args.batch,
            device=args.device,
            workers=4,
            project=args.project,
            name=name,
            seed=42,
            deterministic=True,

            # Repository-inspired, but kept realistic for 4-class detection.
            hsv_h=0.015,
            hsv_s=0.50,
            hsv_v=0.40,
            degrees=8.0,
            translate=0.10,
            scale=0.60,
            fliplr=0.50,
            flipud=0.0,
            perspective=0.0005,
            mosaic=1.0,
            close_mosaic=15,

            # Keep these off in the default baseline so the ablation is interpretable.
            mixup=0.0,
            cutmix=0.0,
        )

        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        best_model = YOLO(str(best))

        # Real-world validation is the primary checkpoint-selection/evaluation view.
        real_metrics = best_model.val(
            data=str(args.data.resolve()),
            split="val",
            imgsz=imgsz,
            device=args.device,
            plots=True,
            verbose=False,
        )

        synth_metrics = best_model.val(
            data=str(synth_diag.resolve()),
            split="val",
            imgsz=imgsz,
            device=args.device,
            plots=False,
            verbose=False,
        )

        row = {
            "name": name,
            "model": model_name,
            "imgsz": imgsz,
            "best_pt": str(best.resolve()),
        }
        row.update({f"real_{k}": v for k, v in extract(real_metrics).items()})
        row.update({f"synth_{k}": v for k, v in extract(synth_metrics).items()})
        rows.append(row)

        print("REAL VAL :", {k: round(v, 4) for k, v in extract(real_metrics).items()})
        print("SYNTH VAL:", {k: round(v, 4) for k, v in extract(synth_metrics).items()})

    out = Path(args.project) / "model_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved comparison table: {out}")
    print("Select primarily by REAL val mAP50-95, then inspect recall, confusion matrix, and latency.")


if __name__ == "__main__":
    main()
