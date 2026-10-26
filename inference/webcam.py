from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=root / "models" / "best.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]

        cv2.imshow("CED Team 5 YOLO", result.plot())
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
