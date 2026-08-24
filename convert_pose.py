#!/usr/bin/env python3
"""
CSV → YOLO-pose 格式：每个目标带 1 个关键点 = 茎点 (StemX, StemY)。

- 类别映射同 detect（作物=0 / 杂草=1），丢弃 label 255 等
- 标签行: cls cx cy w h kx ky 2   (2 = 关键点可见)
- 关键点 = 茎点（归一化到 [0,1]），正是方案2「茎位置切割线」的直接输入
- 与 convert_cropandweed 用相同的 sorted + seed=42 划分 → train/val 完全一致
"""
import random
import shutil
from pathlib import Path

from PIL import Image

import convert_cropandweed as C  # 复用 CROP_IDS / WEED_IDS / map_label / find_image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "yolo_pose"

VAL_RATIO = 0.2
SEED = 42


def main():
    for sub in ("images", "labels"):
        p = OUT / sub
        if p.exists():
            shutil.rmtree(p)
    (OUT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / "val").mkdir(parents=True, exist_ok=True)

    per_image = {}
    dropped = 0
    kept = 0
    for cf in sorted(C.BBOXES.glob("*.csv")):
        stem = cf.stem
        img = C.find_image(stem)
        if img is None:
            continue
        with Image.open(img) as im:
            W, H = im.size
        objs = []
        for line in cf.read_text().strip().splitlines():
            if not line:
                continue
            p = line.split(",")
            if len(p) < 7:
                continue
            left, top, right, bottom, lid, sx, sy = (float(v) for v in p[:7])
            cls = C.map_label(int(lid))
            if cls is None:
                dropped += 1
                continue
            cx = (left + right) / 2 / W
            cy = (top + bottom) / 2 / H
            w = (right - left) / W
            h = (bottom - top) / H
            kx = sx / W
            ky = sy / H
            objs.append((cls, cx, cy, w, h, kx, ky))
            kept += 1
        if objs:
            per_image[stem] = {"img": img, "objs": objs}

    stems = list(per_image.keys())
    random.seed(SEED)
    random.shuffle(stems)
    n_val = int(len(stems) * VAL_RATIO)
    val_stems = set(stems[:n_val])
    train_stems = set(stems[n_val:])

    for split, split_stems in (("train", train_stems), ("val", val_stems)):
        img_dir = OUT / "images" / split
        lbl_dir = OUT / "labels" / split
        for stem in split_stems:
            rec = per_image[stem]
            dst = img_dir / rec["img"].name
            if not dst.exists():
                dst.symlink_to(rec["img"])
            with open(lbl_dir / f"{stem}.txt", "w") as f:
                for (cls, cx, cy, w, h, kx, ky) in rec["objs"]:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kx:.6f} {ky:.6f} 2\n")

    with open(OUT / "data.yaml", "w") as f:
        f.write(f"path: {OUT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n  0: crop\n  1: weed\n")
        f.write("kpt_shape: [1, 3]\n")

    print(f"图像总数: {len(per_image)} | train={len(train_stems)} val={len(val_stems)}")
    print(f"保留目标: {kept} | 丢弃: {dropped}")


if __name__ == "__main__":
    main()
