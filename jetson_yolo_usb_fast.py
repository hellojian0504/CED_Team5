#!/usr/bin/env python3
"""YOLO11n USB camera real-time inference on Jetson (single file + .pt).

Usage:
  python3 jetson_yolo_usb_fast.py
  python3 jetson_yolo_usb_fast.py --camera 0 --imgsz 512
  python3 jetson_yolo_usb_fast.py --no-display

Place best_v3.pt (or best.pt/best_final.pt) in this script's directory.
Requires a CUDA-enabled PyTorch installation, Ultralytics and OpenCV.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class LatestFrameCamera:
    """Read camera continuously; inference always receives the newest frame."""

    def __init__(self, index, width, height, fps):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"USB 카메라 {index}을(를) 열 수 없습니다. "
                "ls /dev/video* 명령으로 장치를 확인하세요."
            )

        # MJPEG can reduce USB bandwidth. These are requests, not guarantees.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ignored by some cameras/backends

        self.frame = None
        self.sequence = 0
        self.stopped = False
        self.failures = 0
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[CAM] /dev/video{index}: {actual_width}x{actual_height}, 요청/보고 FPS={fps}/{actual_fps:.1f}")

    def _read_loop(self):
        while True:
            with self.cond:
                if self.stopped:
                    break
            ok, frame = self.cap.read()
            if not ok:
                self.failures += 1
                if self.failures >= 100:
                    with self.cond:
                        self.stopped = True
                        self.cond.notify_all()
                    break
                time.sleep(0.02)
                continue
            self.failures = 0
            with self.cond:
                self.frame = frame  # cv2.read() returns a new buffer
                self.sequence += 1
                self.cond.notify_all()

    def get(self, previous_sequence, timeout=5.0):
        """Wait until a *new* camera frame exists, then return latest one."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while self.sequence <= previous_sequence and not self.stopped:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, previous_sequence
                self.cond.wait(timeout=remaining)
            if self.sequence <= previous_sequence:
                return None, previous_sequence
            return self.frame, self.sequence

    def close(self):
        with self.cond:
            self.stopped = True
            self.cond.notify_all()
        self.thread.join(timeout=2)
        self.cap.release()


def find_model(folder, explicit):
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"모델 파일이 없습니다: {candidate}")
        return candidate
    for name in ("best_v3.pt", "best.pt", "best_final.pt"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    pts = sorted(folder.glob("*.pt"))
    if len(pts) == 1:
        return pts[0]
    raise FileNotFoundError(
        f"{folder}에 best_v3.pt를 넣어주세요. "
        "여러 .pt 파일이 있다면 --model 경로로 지정하세요."
    )


def main():
    parser = argparse.ArgumentParser(description="Jetson YOLO11n USB Camera fast inference")
    parser.add_argument("--model", type=str, default=None, help=".pt 모델 경로 (기본: 같은 폴더 자동 검색)")
    parser.add_argument("--camera", type=int, default=0, help="USB 카메라 번호 (기본: 0)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 입력 크기 (기본: 640)")
    parser.add_argument("--conf", type=float, default=0.35, help="탐지 신뢰도 (기본: 0.35)")
    parser.add_argument("--width", type=int, default=640, help="카메라 캡처 가로 해상도")
    parser.add_argument("--height", type=int, default=480, help="카메라 캡처 세로 해상도")
    parser.add_argument("--fps", type=int, default=30, help="카메라 요청 FPS")
    parser.add_argument("--no-display", action="store_true", help="화면 출력 없이 최대 속도 측정")
    parser.add_argument("--fp32", action="store_true", help="FP16 비활성화 (디버깅용)")
    parser.add_argument("--allow-cpu", action="store_true", help="CUDA 없을 때 CPU로 실행 허용")
    args = parser.parse_args()

    model_path = find_model(Path(__file__).resolve().parent, args.model)
    cuda = torch.cuda.is_available()
    if not cuda and not args.allow_cpu:
        sys.exit(
            "[오류] CUDA를 사용할 수 없습니다. Jetson용 CUDA 지원 PyTorch가 설치되었는지 "
            "확인하세요. CPU로 시험하려면 --allow-cpu 옵션을 추가하세요."
        )
    device = "cuda:0" if cuda else "cpu"
    use_half = cuda and not args.fp32
    if cuda:
        torch.backends.cudnn.benchmark = True

    print(f"[MODEL] {model_path}")
    print(f"[DEVICE] {device} / FP16={use_half}")
    if cuda:
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print("[INFO] q 또는 ESC로 종료. --no-display로 화면 출력 없이 측정 가능")

    model = YOLO(str(model_path))  # load only once
    cam = LatestFrameCamera(args.camera, args.width, args.height, args.fps)
    try:
        frame, seq = cam.get(0, timeout=10)
        if frame is None:
            raise RuntimeError("카메라에서 프레임을 받지 못했습니다.")

        # First few calls initialize predictor / CUDA kernels.
        with torch.inference_mode():
            for _ in range(3):
                model.predict(
                    source=frame, device=device, imgsz=args.imgsz,
                    half=use_half, conf=args.conf, verbose=False,
                    max_det=30, rect=True,
                )

        last_seq = seq
        n = 0
        last_log = time.perf_counter()
        fps_smoothed = 0.0
        last_end = None
        while True:
            frame, seq = cam.get(last_seq, timeout=5)
            if frame is None:
                raise RuntimeError("카메라 스트림이 중단되었거나 프레임 수신이 지연되었습니다.")
            last_seq = seq

            # Avoid model reloading and video capture latency in the inference loop.
            with torch.inference_mode():
                result = model.predict(
                    source=frame, device=device, imgsz=args.imgsz,
                    half=use_half, conf=args.conf, verbose=False,
                    max_det=30, rect=True,
                )[0]

            now = time.perf_counter()
            if last_end is not None:
                instantaneous = 1.0 / max(now - last_end, 1e-6)
                fps_smoothed = (0.85 * fps_smoothed + 0.15 * instantaneous) if fps_smoothed else instantaneous
            last_end = now
            n += 1

            # YOLO reports timings of preprocessing/inference/postprocessing only.
            ms = result.speed or {}
            infer_ms = ms.get("inference", 0.0)
            if not args.no_display:
                view = result.plot()  # drawing incurs CPU overhead
                cv2.putText(view, f"FPS: {fps_smoothed:.1f}  Infer: {infer_ms:.1f} ms",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Jetson YOLO11n USB", view)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
            if now - last_log >= 2:
                print(f"[RUN] FPS~{fps_smoothed:.1f} | YOLO infer={infer_ms:.1f} ms | detections={len(result.boxes) if result.boxes is not None else 0}")
                last_log = now
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C로 종료")
    finally:
        cam.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
