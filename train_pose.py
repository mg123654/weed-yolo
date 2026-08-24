#!/usr/bin/env python3
"""训练 YOLO11n-pose：杂草/作物框 + 茎点关键点（输出茎切割线用）

用法:
    python train_pose.py               # run 名 = weed_yolo11n_pose, 100 epoch
    python train_pose.py full          # run 名 = weed_yolo11n_pose_full
    python train_pose.py full 30       # 同上, 但 epochs=30 (赶时间/停电前收尾)
"""
import sys

from ultralytics import YOLO


def main():
    name = "weed_yolo11n_pose"
    if len(sys.argv) > 1:
        name = f"weed_yolo11n_pose_{sys.argv[1]}"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    model = YOLO("yolo11n-pose.pt")  # 姿态预训练权重

    results = model.train(
        data="data/yolo_pose/data.yaml",
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
        project="runs",
        name=name,
        patience=20,
        workers=4,
        seed=42,
        verbose=True,
    )
    print("=== pose 训练完成 ===")
    print(f"最佳权重: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
