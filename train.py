#!/usr/bin/env python3
"""训练 YOLO11n 作物/杂草检测 (适配 RTX 3050 Laptop 4GB 显存)

用法:
    python train.py            # run 名 = weed_yolo11n
    python train.py phaseA     # run 名 = weed_yolo11n_phaseA
"""
import sys

from ultralytics import YOLO


def main():
    name = "weed_yolo11n"
    if len(sys.argv) > 1:
        name = f"weed_yolo11n_{sys.argv[1]}"

    model = YOLO("yolo11n.pt")  # COCO 预训练权重, 迁移学习

    results = model.train(
        data="data/yolo/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,          # nano 模型很小, 16 在 4GB 内; 若 OOM 降到 8
        device=0,          # RTX 3050
        project="runs",
        name=name,
        patience=20,       # 早停
        workers=4,
        seed=42,
        verbose=True,
    )
    print("=== 训练完成 ===")
    print(f"最佳权重: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
