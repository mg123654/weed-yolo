#!/usr/bin/env python3
"""
Pose 版评估 + 部署 demo：茎点由模型预测（不再用 GT）。

- 加载 yolo11n-pose 最佳权重 → 预测 作物/杂草框 + 茎点关键点
- 杂草茎点(模型预测) → 切割线段 → 作物 mask(GT) 膨胀 → 消除相交 → 安全切割线
- 评估：框 mAP + 关键点指标 + FPS + 可视化

说明：作物 mask 本阶段仍用 GT（对应 seg 模块，未训练）；茎点已换成模型预测（pose）。
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

import convert_cropandweed as C
from cut_pipeline import compute_safe_cut_lines, draw_results, dilate_mask
from eval_demo import load_crop_mask, load_image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MASKS = DATA / "labelIds" / "CropAndWeed"
OUT = ROOT / "demo_pose"
BEST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "pose" / "runs" / "weed_yolo11n_pose" / "weights" / "best.pt"

MARGIN = 10
EXTEND = 20
MIN_LEN = 5
DEMO_N = 12


def predict(model, img):
    """返回 (所有预测框列表, 杂草框列表, 杂草茎点列表)。茎点为模型预测。"""
    res = model.predict(img, imgsz=640, verbose=False, conf=0.25)[0]
    pred_boxes = []
    weed_boxes = []
    weed_stems = []
    if res.boxes is not None and len(res.boxes):
        boxes = res.boxes
        kps = res.keypoints
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            xyxy = boxes.xyxy[i].cpu().numpy()
            pred_boxes.append((cls, conf, xyxy))
            if cls == 1:  # 杂草
                weed_boxes.append(xyxy)
                kp = kps.xy[i].cpu().numpy()  # (1, 2) 像素坐标
                weed_stems.append((float(kp[0, 0]), float(kp[0, 1])))
    return pred_boxes, weed_boxes, weed_stems


def evaluate(model):
    metrics = model.val(data=str(ROOT / "data" / "yolo_pose" / "data.yaml"),
                        imgsz=640, verbose=False)
    box, pose = metrics.box, metrics.pose
    return {
        "box_map50": float(box.map50),
        "box_map": float(box.map),
        "box_p": float(box.mp),
        "box_r": float(box.mr),
        "pose_map50": float(pose.map50) if pose is not None else None,
        "pose_map": float(pose.map) if pose is not None else None,
        "pose_p": float(pose.mp) if pose is not None else None,
    }


def benchmark_fps(model, n_warmup=10, n_bench=100):
    imgs = sorted((DATA / "images").glob("*.jpg"))
    img = cv2.imread(str(imgs[0])) if imgs else np.zeros((1088, 1920, 3), np.uint8)
    for _ in range(n_warmup):
        model.predict(img, imgsz=640, verbose=False, conf=0.25)
    t0 = time.perf_counter()
    for _ in range(n_bench):
        model.predict(img, imgsz=640, verbose=False, conf=0.25)
    dt = (time.perf_counter() - t0) / n_bench
    return 1.0 / dt, dt * 1000


def demo(model):
    OUT.mkdir(parents=True, exist_ok=True)
    val_imgs = sorted((ROOT / "data" / "yolo_pose" / "images" / "val").glob("*.jpg"))
    picked = val_imgs[:DEMO_N]
    paths = []
    for ip in picked:
        stem = ip.stem
        img = load_image(stem)
        if img is None:
            continue
        pred_boxes, weed_boxes, weed_stems = predict(model, img)
        crop_mask = load_crop_mask(stem)
        if not weed_boxes or crop_mask is None:
            continue
        cuts = compute_safe_cut_lines(weed_boxes, weed_stems, crop_mask,
                                      margin=MARGIN, extend=EXTEND, min_len=MIN_LEN)
        out = draw_results(img, cuts, crop_mask=crop_mask, margin=MARGIN)
        for cls, conf, xyxy in pred_boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{C.CLASS_NAMES[cls]} {conf:.2f}",
                        (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        n_blocked = sum(1 for c in cuts if c["blocked"])
        cv2.putText(out, f"weed={len(weed_boxes)} blocked={n_blocked} (stems=pose-pred)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        dst = OUT / f"{stem}.jpg"
        cv2.imwrite(str(dst), out)
        paths.append(str(dst))
    return paths


def main():
    model = YOLO(str(BEST))
    print("=== ① pose 评估 ===")
    m = evaluate(model)
    print(json.dumps(m, indent=2, ensure_ascii=False))

    print("\n=== ② FPS ===")
    fps, ms = benchmark_fps(model)
    print(f"FPS = {fps:.2f}  ( {ms:.1f} ms/帧 )")

    print("\n=== ③ demo（茎点=模型预测）===")
    paths = demo(model)
    print(f"生成 {len(paths)} 张 → demo_pose/")

    summary = {"eval": m, "fps": round(fps, 2), "ms_per_frame": round(ms, 1),
               "demo_images": paths, "note": "茎点=pose模型预测; 作物mask仍用GT(seg待训)"}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("写入 demo_pose/summary.json")


if __name__ == "__main__":
    main()
