#!/usr/bin/env python3
"""
评估 + 部署 demo（mAP / FPS / 作物避让可视化）

输入：
  - runs/weed_yolo11n/weights/best.pt  （训练好的 YOLO11n detect 模型）
  - data/yolo/                         （转换后的 YOLO 格式）
  - data/bboxes/*.csv                  （GT 框 + 茎点，用于 demo 的杂草茎/框）
  - data/labelIds/CropAndWeed/*.png    （GT 语义 mask，用于作物避让禁区）

产出：
  - 控制台打印 mAP@0.5 / mAP@0.5:0.95 / FPS
  - demo/*.jpg                        （叠加预测框 + 安全切割线的可视化）
  - demo/summary.json                 （量化结果汇总）

说明（诚实标注）：
  - 检测（杂草/作物框）来自已训练的 detect 模型。
  - 茎点 + 作物 mask 目前用 GT 代替（对应方案里的 pose / seg 两个模块，尚未训练），
    目的是在本阶段就把「安全切割线 + 作物避让」这条部署链路端到端跑通。
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

import convert_cropandweed as C  # 复用 CROP_IDS / WEED_IDS / map_label
from cut_pipeline import compute_safe_cut_lines, draw_results, dilate_mask

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BBOXES = DATA / "bboxes"
MASKS = DATA / "labelIds" / "CropAndWeed"
OUT = ROOT / "demo"
# 权重路径可传参: python eval_demo.py runs/xxx/weights/best.pt
BEST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "weed_yolo11n" / "weights" / "best.pt"

MARGIN = 10      # 作物 mask 膨胀像素（安全余量）
EXTEND = 20      # 切割线在茎两侧延长像素
MIN_LEN = 5      # 最短可打线段
DEMO_N = 12      # 出多少张可视化


def load_weed_boxes_stems(stem: str):
    """从 GT bbox CSV 读杂草的框(像素) + 茎点(像素)。返回 (boxes, stems)。"""
    boxes, stems = [], []
    cf = BBOXES / f"{stem}.csv"
    if not cf.exists():
        return boxes, stems
    for line in cf.read_text().strip().splitlines():
        if not line:
            continue
        p = line.split(",")
        if len(p) < 7:
            continue
        left, top, right, bottom, lid, sx, sy = (float(v) for v in p[:7])
        if C.map_label(int(lid)) == 1:  # 只取杂草
            boxes.append((left, top, right, bottom))
            stems.append((sx, sy))
    return boxes, stems


def load_crop_mask(stem: str):
    """从 GT 语义 mask 构建作物二值 mask（LabelID ∈ CROP_IDS → 作物）。"""
    mf = MASKS / f"{stem}.png"
    if not mf.exists():
        return None
    sem = np.array(Image.open(mf))
    return np.isin(sem, np.array(list(C.CROP_IDS), dtype=np.uint8)).astype(np.uint8)


def load_image(stem: str):
    p = DATA / "images" / f"{stem}.jpg"
    if not p.exists():
        return None
    return cv2.imread(str(p))


def evaluate():
    """mAP + 官方速度指标。"""
    model = YOLO(str(BEST))
    metrics = model.val(data=str(ROOT / "data" / "yolo" / "data.yaml"),
                        imgsz=640, verbose=False)
    box = metrics.box
    return {
        "map50": float(box.map50),
        "map": float(box.map),
        "map75": float(box.map75),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }


def benchmark_fps(n_warmup=10, n_bench=100):
    """单卡 batch=1 实测 FPS（用真实 1088p 图走 640 letterbox）。"""
    model = YOLO(str(BEST))
    # 找一张真实 val 图
    imgs = sorted((ROOT / "data" / "images").glob("*.jpg"))
    img = cv2.imread(str(imgs[0])) if imgs else np.zeros((1088, 1920, 3), np.uint8)
    for _ in range(n_warmup):
        model.predict(img, imgsz=640, verbose=False, conf=0.25)
    t0 = time.perf_counter()
    for _ in range(n_bench):
        model.predict(img, imgsz=640, verbose=False, conf=0.25)
    dt = (time.perf_counter() - t0) / n_bench
    return 1.0 / dt, dt * 1000  # fps, ms/帧


def demo_cutlines():
    """在若干 val 图上：预测框 + GT茎点/作物mask → 安全切割线 → 可视化。"""
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BEST))
    # 用 val 集的图片（软链接目录里），按文件名取 stem
    val_imgs = sorted((ROOT / "data" / "yolo" / "images" / "val").glob("*.jpg"))
    picked = val_imgs[:DEMO_N]
    results_paths = []
    for ip in picked:
        stem = ip.stem
        img = load_image(stem)
        if img is None:
            continue
        # ① 模型预测框
        pred = model.predict(img, imgsz=640, verbose=False, conf=0.25)[0]
        # ② GT 杂草框 + 茎点 + 作物 mask
        weed_boxes, stems = load_weed_boxes_stems(stem)
        crop_mask = load_crop_mask(stem)
        if not weed_boxes or crop_mask is None:
            continue
        # ③ 安全切割线
        cuts = compute_safe_cut_lines(weed_boxes, stems, crop_mask,
                                      margin=MARGIN, extend=EXTEND, min_len=MIN_LEN)
        # ④ 可视化：先画避让后处理（茎点+安全线+作物禁区），再叠预测框
        out = draw_results(img, cuts, crop_mask=crop_mask, margin=MARGIN)
        for box in pred.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # 作物绿 / 杂草红
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{C.CLASS_NAMES[cls]} {conf:.2f}",
                        (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        n_blocked = sum(1 for c in cuts if c["blocked"])
        cv2.putText(out, f"weed={len(weed_boxes)} blocked={n_blocked}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        dst = OUT / f"{stem}.jpg"
        cv2.imwrite(str(dst), out)
        results_paths.append(str(dst))
    return results_paths


def main():
    print("=== ① 评估 mAP ===")
    m = evaluate()
    print(json.dumps(m, indent=2, ensure_ascii=False))

    print("\n=== ② 实测 FPS ===")
    fps, ms = benchmark_fps()
    print(f"FPS = {fps:.2f}  ( {ms:.1f} ms/帧 )  [batch=1, imgsz=640, RTX3050]")

    print("\n=== ③ 作物避让 demo ===")
    paths = demo_cutlines()
    print(f"生成 {len(paths)} 张可视化图 → demo/")

    summary = {"map": m, "fps": round(fps, 2), "ms_per_frame": round(ms, 1),
               "demo_images": paths, "note": "茎点+作物mask当前用GT代替(pose/seg模块待训)"}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n结果已写入 demo/summary.json")


if __name__ == "__main__":
    main()
