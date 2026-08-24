# 杂草检测与激光除草系统

> 农田场景下**实时检测杂草 → 输出茎部切割线 → 激光精准除草**，配合作物轮廓避让后处理，保证**不误伤农作物**。这是激光除草机器人的「感知 + 安全决策」模块。

## 一句话方案

**模型只负责「看见」**（杂草框 + 茎点 + 作物轮廓）；**「哪里不能打」的决策交给确定性代码**（mask 膨胀 + 相交消除），因为「误伤作物」是不可接受的失败模式，不能容忍神经网络的一次幻觉。

## 总体流水线

```
输入图像
   │
   ├─ ① 检测（打哪里）：YOLO11 detect / pose → 杂草框 + 茎点
   ├─ ② 避让（不能打哪）：YOLO11 seg → 作物轮廓 mask
   ▼
安全后处理（纯代码 cut_pipeline.py）
   1. 茎点 → 水平短线段（40 px，不横跨整框）
   2. 作物 mask 膨胀 N 像素（预留安全余量）
   3. 线段 ∩ 膨胀后 mask → 消除相交子段
   ▼
输出：安全切割线段（JSON 文本 → 激光 / 运动控制模块）
```

## 实验结果（CropAndWeed 全量 7705 图）

| 模型 | 指标 | 值 |
|---|---|---|
| **detect**（YOLO11n，100 ep） | mAP@0.5 / 0.5:0.95 | **0.808** / **0.569** |
| | FPS | **99.4**（10.1 ms/帧） |
| **pose**（YOLO11n-pose，30 ep） | 茎点 mAP@0.5 / 0.5:0.95 | **0.841** / **0.835** |
| | FPS | **99.6**（10.0 ms/帧） |

> 帧率目标 ≥ 30 FPS，实测约 **3 倍余量**。完整方案与测试样例见 [`report/design_report.pdf`](report/design_report.pdf)。

## 目录结构

```
weed-yolo/
├── convert_cropandweed.py   # CSV 标注 → YOLO 格式（作物 0 / 杂草 1，茎点存 stems.json）
├── convert_pose.py          # CSV → YOLO-pose 格式（每目标 1 关键点 = 茎点）
├── train.py                 # YOLO11n detect 训练
├── train_pose.py            # YOLO11n-pose 训练
├── eval_demo.py             # detect 评估 + FPS + 可视化
├── eval_demo_pose.py        # pose 评估 + 可视化（茎点 = 模型预测）
├── cut_pipeline.py          # 作物避让 + 安全切割线（部署核心后处理）
├── data/yolo/data.yaml      # detect 数据配置
├── data/yolo_pose/data.yaml # pose 数据配置
├── PLAN.md                  # 方案设计文档
├── LOG.md                   # 执行步骤记录 / 运行日志
└── report/                  # 方案设计报告（LaTeX 源码 + PDF + 样例图）
```

> 训练数据（`data/` 图片与标注）、模型权重（`runs/`、`*.pt`）、虚拟环境（`venv/`）**不入库**，见 `.gitignore`。

## 快速开始

### 环境

```
torch 2.13.0+cu130   ultralytics 8.4.127   opencv-python 5.0.0
```

### 数据准备

1. 下载 [CropAndWeed](https://github.com/cropandweed/cropandweed-dataset)（8034 图 + 标注，非商用许可）
2. `python convert_cropandweed.py`  → 生成 `data/yolo/`
3. `python convert_pose.py`        → 生成 `data/yolo_pose/`

### 训练

```bash
python train.py full              # YOLO11n detect
python train_pose.py full 30      # YOLO11n-pose（茎点）
```

### 评估 + 可视化

```bash
python eval_demo.py      runs/detect/runs/weed_yolo11n_full/weights/best.pt
python eval_demo_pose.py runs/pose/runs/weed_yolo11n_pose_full/weights/best.pt
```

## 输出接口（下游模块只消费文本）

```json
{"frame_id": 1001, "num_weeds": 2, "num_blocked": 0,
 "weeds": [
   {"id": 0, "stem": [11.0, 118.0],
    "cut_segments": [[0, 31, 118]], "blocked": false},
   {"id": 1, "stem": [724.0, 1079.0],
    "cut_segments": [[704, 744, 1079]], "blocked": false}]}
```

- `stem`：茎点（离散点）
- `cut_segments`：安全切割线段 `[x0, x1, y]`（激光要扫的线）
- `blocked`：是否被作物完全挡住（挡住则跳过该株）

当前为**像素坐标**；接入深度相机 + 手眼标定后升级为机器人基座系三维坐标。

## 许可

- 数据集 **CropAndWeed**：非商用许可，仅限研究 / 实验。
- 本项目代码：待定。
