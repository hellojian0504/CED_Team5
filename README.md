# CED Team 5 — YOLO Object Detection

4-class YOLO11 project for the CED Team 5 robot.

Classes:
- 0: Scylla
- 1: TrojanHorse
- 2: cube_10cm
- 3: octahedron_10cm

## Structure

```text
CED_Team5/
├─ train/       # dataset build, training, fine-tuning, evaluation
├─ inference/   # image/video/webcam inference + Jetson TensorRT export
├─ models/      # put only the final selected model here
├─ requirements.txt
└─ .gitignore
```

Datasets, `runs/`, temporary checkpoints, and TensorRT `.engine` files are intentionally not stored in Git.

## Install

```bash
pip install -r requirements.txt
```

## Train

CPU example:

```bash
python train/train_grid.py --data dataset/data.yaml --device cpu --batch 4 --epochs 180
```

GPU example:

```bash
python train/train_grid.py --data dataset/data.yaml --device 0 --batch 4 --epochs 180
```

## Fine-tune on real data

```bash
python train/finetune_real.py --weights models/best.pt --data dataset/data_real_finetune.yaml --device 0 --batch 4 --imgsz 640
```

## Evaluate

```bash
python train/evaluate_model.py --weights models/best.pt --data dataset/data.yaml --split test --imgsz 640 --device cpu
```

## Webcam

PC / CPU:

```bash
python inference/webcam.py --weights models/best.pt --device cpu
```

Jetson:

```bash
python inference/webcam.py --weights models/best.pt --device 0
```

Press `q` or `Esc` to quit.

## Jetson TensorRT

On the Jetson itself:

```bash
python inference/export_tensorrt.py --weights models/best.pt --imgsz 640
```

This creates a TensorRT engine for that Jetson environment. The generated `.engine` file is not committed.

## Model file

After training/fine-tuning is complete, copy the final model to:

```text
models/best.pt
```

Keep only the final selected `best.pt` in Git. Do not commit datasets, every epoch checkpoint, or the whole `runs/` folder.
