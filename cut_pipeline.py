#!/usr/bin/env python3
"""
作物避让 + 安全切割线 —— 后处理模块（部署方案核心）

流水线（纯代码，非模型）：
    杂草框 + 茎位置
        → ① 生成切割线段（含延长余量，吸收茎细/摆动/抖动误差）
        → ② 作物 mask 膨胀一圈（预留安全余量）
        → ③ 切割线段 ∩ 膨胀后的 mask → 消除相交段
        → 输出安全切割线段

关键原则：框只负责「定位杂草在哪」，避让决策用「作物 mask（膨胀后）」，不用框。
"""

import numpy as np
import cv2


def dilate_mask(crop_mask: np.ndarray, margin: int = 10) -> np.ndarray:
    """对作物二值 mask 膨胀 margin 像素, 作为安全禁区.

    margin = 激光光斑半径 + 安全距离 + 分割边缘误差 + 茎摆动余量.
    """
    if crop_mask is None or crop_mask.size == 0:
        return None
    # 3x3 核迭代 margin 次 = 精确向外膨胀 margin 像素
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(crop_mask.astype(np.uint8), kernel, iterations=margin)


def cut_segment(stem_x: float, stem_y: float, half_len: float):
    """由茎点生成水平切割线段（含延长余量）.

    返回 (起点, 终点), 水平线, 覆盖茎宽 ± half_len.
    """
    x0 = int(round(stem_x - half_len))
    x1 = int(round(stem_x + half_len))
    y = int(round(stem_y))
    return (x0, y), (x1, y)


def safe_segments(seg_start, seg_end, crop_mask_dilated, min_len: int = 5):
    """把一条水平切割线段按「是否落在作物 mask 内」切成若干安全子段.

    返回 [(起x, 终x, y), ...] 中, 每个子段都不与作物 mask 相交.
    """
    (x0, y), (x1, y) = seg_start, seg_end
    if x1 < x0:
        x0, x1 = x1, x0
    if crop_mask_dilated is None:
        return [(x0, x1, y)] if x1 - x0 >= min_len else []

    H, W = crop_mask_dilated.shape
    x0 = max(0, min(x0, W - 1))
    x1 = max(0, min(x1, W - 1))
    y = max(0, min(y, H - 1))

    # 沿线取 mask 值, 标记"禁区"
    xs = np.arange(x0, x1 + 1)
    forbidden = crop_mask_dilated[y, xs] > 0

    segments = []
    start = None
    for i, f in enumerate(forbidden):
        if not f and start is None:
            start = xs[i]
        elif f and start is not None:
            if xs[i - 1] - start >= min_len:
                segments.append((int(start), int(xs[i - 1]), y))
            start = None
    if start is not None and xs[-1] - start >= min_len:
        segments.append((int(start), int(xs[-1]), y))
    return segments


def compute_safe_cut_lines(
    weed_boxes,
    stem_points,
    crop_mask,
    margin: int = 10,
    extend: int = 20,
    min_len: int = 5,
):
    """端到端: 输入杂草框 + 茎点 + 作物mask, 输出安全切割线段列表.

    weed_boxes:   [(x1, y1, x2, y2), ...] 像素坐标
    stem_points:  [(sx, sy), ...] 与 weed_boxes 一一对应
    crop_mask:    作物轮廓二值 mask (H, W), 可为 None 表示"无作物/无避让"
    margin:       作物 mask 膨胀像素(安全余量)
    extend:       切割线在茎点两侧的延长像素(吸收定位误差)

    返回 list of dict: {"weed_idx", "stem", "segment" (起x,终x,y), "blocked" (是否被完全挡住)}
    """
    crop_dilated = dilate_mask(crop_mask, margin) if crop_mask is not None else None
    results = []
    for i, (box, stem) in enumerate(zip(weed_boxes, stem_points)):
        sx, sy = stem
        # 短线段: 半长 = extend(茎点两侧固定余量), 覆盖茎宽 + 定位/摆动误差.
        # 不能用框宽做半长 —— 框宽是整株冠幅(上百像素), 茎只有几个像素,
        # 否则切割线会横跨整个框(违反方案2改良版「短线段不横跨整框」)。
        half_len = extend
        seg_start, seg_end = cut_segment(sx, sy, half_len)
        segs = safe_segments(seg_start, seg_end, crop_dilated, min_len=min_len)
        results.append({
            "weed_idx": i,
            "box": box,
            "stem": (float(sx), float(sy)),
            "segments": segs,                    # 安全可打线段
            "blocked": len(segs) == 0,           # 是否被作物完全挡住
        })
    return results


def cuts_to_text(results, image_size=None, frame_id=None, timestamp_ms=None,
                 coordinate_space="pixel"):
    """把 compute_safe_cut_lines 的结构化结果序列化为 JSON 文本, 供下游模块消费.

    下游(激光控制/运动控制)只关心坐标, 不关心图像. 每个 weed 一条:
      - stem          茎点(离散点)
      - cut_segments  安全切割线段 [(x0, x1, y), ...] —— 激光要扫的线
      - blocked       是否被作物完全挡住(挡住则跳过该株)

    坐标默认"像素"; 手眼标定后把 coordinate_space 换成 "world_mm" 并附标定矩阵即可.
    """
    import json
    weeds = []
    for r in results:
        weeds.append({
            "id": r["weed_idx"],
            "stem": [round(float(r["stem"][0]), 2), round(float(r["stem"][1]), 2)],
            "cut_segments": [[int(a), int(b), int(y)] for (a, b, y) in r["segments"]],
            "blocked": r["blocked"],
        })
    msg = {
        "coordinate_space": coordinate_space,
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "image_size": image_size,
        "num_weeds": len(weeds),
        "num_blocked": sum(1 for w in weeds if w["blocked"]),
        "weeds": weeds,
    }
    return json.dumps(msg, ensure_ascii=False)


def cuts_to_lines(results):
    """轻量行协议(每株一行), 便于串口/socket 直接发. blocked 的株输出 BLOCKED."""
    lines = []
    for r in results:
        sx, sy = r["stem"]
        segs = r["segments"]
        if not segs:
            lines.append(f"W {r['weed_idx']} stem=({sx:.1f},{sy:.1f}) BLOCKED")
        else:
            for (x0, x1, y) in segs:
                lines.append(f"W {r['weed_idx']} stem=({sx:.1f},{sy:.1f}) cut=({x0},{y})->({x1},{y})")
    return "\n".join(lines)


def draw_results(img, results, crop_mask=None, margin=10):
    """可视化: 画作物mask禁区、茎点、安全切割线段. 返回标注后的 BGR 图."""
    out = img.copy()
    if crop_mask is not None:
        d = dilate_mask(crop_mask, margin)
        overlay = np.zeros_like(out)
        overlay[:, :] = (0, 0, 255)  # 红色禁区
        out = cv2.addWeighted(out, 1.0, cv2.bitwise_and(overlay, overlay, mask=d), 0.35, 0)
    for r in results:
        x1, y1, x2, y2 = r["box"]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        cv2.circle(out, (int(r["stem"][0]), int(r["stem"][1])), 5, (0, 255, 0), -1)
        for (sx0, sx1, y) in r["segments"]:
            cv2.line(out, (sx0, y), (sx1, y), (0, 255, 0), 4)  # 绿色=安全可打
    return out
