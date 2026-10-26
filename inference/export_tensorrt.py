from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=root / "models" / "best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    output = model.export(
        format="engine",
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
    )
    print("TensorRT engine:", output)


if __name__ == "__main__":
    main()
