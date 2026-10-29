
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
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    frame_id = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1

            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]

            # 프레임마다 탐지 결과 출력
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                labels = []

                for box in boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    label = result.names[class_id]

                    labels.append(f"{label} ({confidence:.2f})")

                print(
                    f"[Frame {frame_id}] "
                    f"Detected {len(boxes)}: "
                    + ", ".join(labels),
                    flush=True
                )
            else:
                print(
                    f"[Frame {frame_id}] No detection",
                    flush=True
                )

            cv2.imshow("CED Team 5 YOLO", result.plot())

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

