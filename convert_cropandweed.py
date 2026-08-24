#!/usr/bin/env python3
"""
把 CropAndWeed 的 bbox CSV (Left,Top,Right,Bottom,LabelID,StemX,StemY) 转成 YOLO 检测格式。

- 类别映射采用 CropOrWeed2: 作物=0, 杂草=1
- 丢弃 label 255 (Vegetation 兜底类) 及未列入 crop/weed 的类别
- 同时把每个目标的茎点 (StemX, StemY, 归一化) 存到 stems.json, 留给后续茎线任务
- 图片用软链接(不复制 11GB), 按图像划分 train/val (避免目标级泄漏)
"""
import csv
import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BBOXES = DATA / "bboxes"
IMAGES = DATA / "images"
OUT = DATA / "yolo"

VAL_RATIO = 0.2
SEED = 42

# CropOrWeed2 映射 (来自 cnw/utilities/datasets.py)
CROP_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 94, 24, 18, 13, 26, 27, 15}
WEED_IDS = {
    31, 48, 62, 65, 68, 69, 74, 75, 81, 84, 86, 32, 29, 33, 37, 49, 30, 44, 66,
    87, 89, 91, 61, 79, 34, 41, 52, 35, 36, 78, 38, 39, 71, 72, 88, 42, 45, 70,
    47, 51, 54, 58, 60, 80, 83, 96, 22, 63, 85, 56, 57, 64, 77, 50, 59, 67, 76,
}

CLASS_NAMES = {0: "crop", 1: "weed"}


def map_label(lid: int):
    if lid in CROP_IDS:
        return 0
    if lid in WEED_IDS:
        return 1
    return None  # 丢弃: 255 Vegetation + 其他未列入的作物


def find_image(stem: str):
    for ext in (".jpg", ".png", ".jpeg"):
        p = IMAGES / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    # 清理旧输出，避免子集/全量重跑时软链接残留导致图片同时出现在两个 split
    for sub in ("images", "labels"):
        p = OUT / sub
        if p.exists():
            shutil.rmtree(p)
    (OUT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / "val").mkdir(parents=True, exist_ok=True)

    csv_files = sorted(BBOXES.glob("*.csv"))
    per_image = {}  # stem -> {"img": Path, "objs": [ {cls, cx, cy, w, h, sx, sy, orig} ]}
    missing_imgs = []
    dropped_objs = 0
    kept_objs = 0

    for cf in csv_files:
        stem = cf.stem
        img = find_image(stem)
        if img is None:
            missing_imgs.append(stem)
            continue
        with Image.open(img) as im:
            W, H = im.size
        objs = []
        with open(cf, newline="") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                left, top, right, bottom, lid, sx, sy = parts[:7]
                lid = int(lid)
                cls = map_label(lid)
                if cls is None:
                    dropped_objs += 1
                    continue
                left, top, right, bottom = (float(v) for v in (left, top, right, bottom))
                cx = (left + right) / 2 / W
                cy = (top + bottom) / 2 / H
                w = (right - left) / W
                h = (bottom - top) / H
                objs.append({
                    "cls": cls,
                    "cx": cx, "cy": cy, "w": w, "h": h,
                    "sx": float(sx) / W, "sy": float(sy) / H,
                    "orig_label": lid,
                })
                kept_objs += 1
        if objs:
            per_image[stem] = {"img": img, "objs": objs}

    # 按图像划分
    stems = list(per_image.keys())
    random.seed(SEED)
    random.shuffle(stems)
    n_val = int(len(stems) * VAL_RATIO)
    val_stems = set(stems[:n_val])
    train_stems = set(stems[n_val:])

    for split, split_stems in (("train", train_stems), ("val", val_stems)):
        img_dir = OUT / "images" / split
        lbl_dir = OUT / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for stem in split_stems:
            rec = per_image[stem]
            # 软链接图片
            dst = img_dir / rec["img"].name
            if not dst.exists():
                os.symlink(rec["img"], dst)
            # 写 YOLO 标签
            with open(lbl_dir / f"{stem}.txt", "w") as f:
                for o in rec["objs"]:
                    f.write(f"{o['cls']} {o['cx']:.6f} {o['cy']:.6f} {o['w']:.6f} {o['h']:.6f}\n")

    # 茎点侧输出
    stems_out = {}
    for stem, rec in per_image.items():
        stems_out[stem] = [{"cls": o["cls"], "sx": o["sx"], "sy": o["sy"], "orig_label": o["orig_label"]}
                           for o in rec["objs"]]
    with open(OUT / "stems.json", "w") as f:
        json.dump(stems_out, f)

    # data.yaml
    with open(OUT / "data.yaml", "w") as f:
        f.write(f"path: {OUT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write("  0: crop\n")
        f.write("  1: weed\n")

    print(f"图像总数(有图): {len(per_image)} | train={len(train_stems)} val={len(val_stems)}")
    print(f"保留目标: {kept_objs} | 丢弃目标(255等): {dropped_objs}")
    print(f"缺失图片(未下载): {len(missing_imgs)}")
    if missing_imgs[:5]:
        print("  示例:", missing_imgs[:5])


if __name__ == "__main__":
    main()
