# 执行步骤记录 / 运行日志

> 项目：杂草检测 + 激光除草
> 目标：全自动完成 数据下载 → 环境 → 转换 → 训练 → 评估 → 部署（作物避让后处理）
> 开始时间：2026-08-24

---

## 目录结构（本工程）

```
weed-yolo/
├── PLAN.md                    # 方案设计文档
├── LOG.md                     # 本步骤记录
├── cropandweed-dataset/       # 官方仓库(脚本/类别映射参考)
├── convert_cropandweed.py     # 标注 → YOLO 格式转换
├── train.py                   # YOLO11n 训练脚本
├── cut_pipeline.py            # 作物避让 + 安全切割线 后处理模块
├── data/                      # 数据(下载中)
│   ├── bboxes/                # 8034 个 CSV (框+茎点)
│   ├── labelIds/CropAndWeed/  # 分割 mask
│   ├── params/                # 元数据
│   ├── images/                # 图片(下载中)
│   └── yolo/                  # 转换后的 YOLO 格式(待生成)
├── runs/                      # 训练输出(待生成)
└── venv/                      # Python 环境(安装中)
```

---

## 步骤记录

### [已完成] Step 0 — 方案定稿
- 详见 `PLAN.md`：YOLO11 选型、方案2改良版（短线段+作物避让）、mask 避让(L2)、CropAndWeed 数据集。

### [已完成] Step 1 — 数据集选型与标注下载
- 选定 **CropAndWeed**（WACV 2023，8034 图，1920×1088 RGB）。
- 已下载 `cropandweed_annotations.tar`（105MB）并解压：
  - `data/bboxes/`：8034 个 CSV，格式 `Left,Top,Right,Bottom,LabelID,StemX,StemY`
  - `data/labelIds/CropAndWeed/`：语义分割 mask
  - `data/params/`：8034 个元数据 CSV
- 关键发现：**label 255（Vegetation 兜底类）3.3 万实例 ≈ 总目标 40%**，转换时丢弃。

### [已完成] Step 2 — 图片下载
- 4 个 tar 包共 10.6GB（images1~4of4，实测单个 2.58~3.0GB）。
- ✅ **8034 张图全部下完并解压**（比预估快，约 1 小时内完成）。

### [已完成] Step 3 — 环境安装（uv venv + CUDA torch）
- `uv venv` + `torch` + `ultralytics` 安装完成。
- 遇到 `nvidia-nvjitlink-cu12`（pypi.nvidia.com）超时，uv 自动重试后成功。
- ✅ 验证通过：`torch 2.13.0+cu130 | cuda: True | NVIDIA GeForce RTX 3050 Laptop GPU`
- ✅ `ultralytics 8.4.127`、`opencv 5.0.0`

### [已完成] Step 4 — 脚本准备 + 单元自测
- `convert_cropandweed.py`：CSV → YOLO 格式（作物=0/杂草=1，茎点存 stems.json，按图划分 train/val）。
- `train.py`：YOLO11n，imgsz 640，batch 16，100 epochs，早停 20。
- `cut_pipeline.py`：茎点→切割线段→作物mask膨胀→相交消除→安全切割线段。
- `eval_demo.py`：评估 mAP + 实测 FPS + 作物避让可视化（新增，见 Step 7/8）。
- ✅ `cut_pipeline.py` 合成数据自测通过：
  - 作物 mask(80~120) + margin=10 → 禁区精确为 70~130
  - 左侧杂草 → 安全线段 (5,45)；中间压作物杂草 → 被切为 (51,69)+(130,159) 两段；右侧 → (125,196)
  - 语义修正：`cv2.dilate` 改用 3×3 核 + iterations=margin，实现**精确 margin 像素膨胀**
- ✅ 类别映射全量统计：8034 张图中 **7705 张含作物/杂草**（进入训练），**329 张全为 label 255**（Vegetation 兜底，正确丢弃）。GT mask 为 1920×1088 语义 mask（像素=LabelID），避让时 `np.isin(mask, CROP_IDS)` 转作物二值 mask。

### [进行中] 执行策略（两阶段，因下载 9 小时是瓶颈）
1. **阶段 A（先行验证）**：第 1 个 tar 解压出 ~2000 图后，先跑一轮「转换→训练→评估→demo」，端到端打通并产出首批结果。
2. **阶段 B（最终版）**：全量 7705 图下完后，重跑转换 + 完整训练（YOLO11n 训练仅约 30 分钟），出最终 mAP / FPS / demo。

### [阶段A进行中] Step 5 — 转换（子集先行）
- 第 1 个 tar 解压出 **2003 张图**（18:07 完成），转换结果：
  - **有效图 1739 张**（train=1392 / val=347，按图划分 seed=42）
  - 保留目标 **19785**（作物 4149 / 杂草 11574，约 2.8:1），丢弃 label 255 共 8049
  - 数据源：其余 6031 张图仍在后台下载（tar 2/3/4）

### [阶段A✅] Step 6 — 训练
- `venv/bin/python train.py phaseA`（后台）→ run 名 `weed_yolo11n_phaseA`
- YOLO11n（182 层 / 2.59M 参数 / 6.5 GFLOPs），100 epochs，imgsz 640，batch 16，早停 20。
- 结果：**100 epochs / 31 分钟**（0.517h），batch 16 未 OOM（显存峰值 3.1GB/4GB），最佳权重 `runs/detect/runs/weed_yolo11n_phaseA/weights/best.pt`。

### [阶段A✅] Step 7 — 评估
| 指标 | 值 |
|---|---|
| **mAP@0.5** | **0.758** |
| **mAP@0.5:0.95** | **0.515** |
| mAP@0.75 | 0.540 |
| Precision | 0.764 |
| Recall | 0.720 |
| **FPS** | **99.7**（10ms/帧，batch=1，imgsz 640） |

- FPS 远超 30 目标（约 3 倍余量）；mAP@0.5 0.758 对「子集 1739 图 + 丢弃 40% 模糊目标」是扎实的地基。

### [阶段A✅] Step 8 — 部署验证
- `eval_demo.py` 在 12 张 val 图上跑通：预测框 + GT茎点/作物mask → 作物避让 → 安全切割线。
- 产出 **9 张可视化**（`demo/*.jpg`）：绿色=安全可打线段，红色=作物膨胀禁区，黄框=作物，红框=杂草。
- ✅ 避让逻辑在真实图上正确：样例中每株杂草输出 1 条安全线段（此批无作物遮挡）；已验证存在被作物挡住时会正确切分/判 blocked。
- 诚实标注：本阶段茎点 + 作物 mask 用 **GT 代替**（对应方案里的 pose / seg 两个模块，尚未训练）。

### [已完成] Step 8.5 — 修复「切割线横跨整框」bug
- **现象**：demo 里杂草的切割短线横跨整个框，违背方案2改良版「短线段不横跨整框」。
- **根因**：`cut_pipeline.py` 中 `half_len = max(extend, 0.6×框宽)`，把「框宽（整株冠幅，上百像素）」误当成「茎宽（几个像素）」，导致切割线总长 = 1.2×框宽。
- **修复**：改为 `half_len = extend`（固定 20px → 茎两侧各 20px 共 40px 短线），不再依赖框宽。
- **验证**（纯 CPU，GT 框+茎+mask）：修复后 40px 短线，即便框宽 432px 也仍是 40px；伸进作物禁区的短线被正确裁短（33/24/30px 等），避让逻辑正常。
- 最终 demo（含模型预测框）待全量训练完成后用修复版重出。

---

## 结果（阶段 A · 子集 1739 图先行验证）

> 数据：子集 1739 张有效图（train 1392 / val 347，~2 万目标）
> 模型：YOLO11n（2.59M 参数），100 epochs，imgsz 640，batch 16

| 指标 | 值 | 备注 |
|---|---|---|
| mAP@0.5 | **0.758** | 杂草/作物二分类 |
| mAP@0.5:0.95 | **0.515** | |
| mAP@0.75 | 0.540 | |
| Precision / Recall | 0.764 / 0.720 | |
| **FPS** | **99.7** | batch=1，10ms/帧，≥30 达标（3× 余量） |

**产物**：
- 权重：`runs/detect/runs/weed_yolo11n_phaseA/weights/best.pt`（5.5MB）
- 可视化：`demo/*.jpg`（9 张，预测框 + 作物禁区 + 安全切割线）
- 指标 JSON：`demo/summary.json`

**结论**：
1. 检测地基成立 —— 子集上 mAP@0.5 达 0.76，能稳定区分杂草/作物。
2. 实时性达标 —— 99.7 FPS，即使叠加作物避让后处理（numpy/cv2，微秒级）仍远高于 30。
3. 部署链路（检测 → 茎线 → 作物避让 → 安全切割线）端到端跑通，避让逻辑在真实图验证正确。

**阶段 B（全量重训，✅ 已完成）**：
- ✅ 8034 图全量下完 → 重转 detect + pose：**7705 张**（train 6164 / val 1541），78288 目标。
- ✅ 新增 `convert_pose.py`（CSV→YOLO-pose，每目标 1 关键点=茎点，kpt_shape=[1,3]）、`train_pose.py`（支持 argv 传 epochs）、`eval_demo_pose.py`（茎点换模型预测）。
- ✅ 全量 detect 训练完成（`train.py full` → `weed_yolo11n_full`，100 epoch）。
- ✅ 全量 pose 训练完成（`train_pose.py full 30` → `weed_yolo11n_pose_full`，30 epoch，停电前收尾版；`last.pt` 已落盘，可 `resume` 续训至 100）。
- ⚠️ **茎点标注质量发现**：抽查 200 图 / 3440 目标，**69% 的 StemX/StemY ≈ 框中心**（偏移<2px），中位偏移 0.7px。即 CropAndWeed 的「茎点」多为近似植物中心、非精确茎基部。pose 学到的更像「框中心」，真机要精确切茎基部需自标「真茎基部」关键点做微调。
- ⬜ 补齐 seg（作物 mask）→ 把 demo 里 GT mask 也换成模型预测，完成真正端到端（后续）。

---

## 最终结果（阶段 B · 全量 7705 图）

| 模型 | 指标 | 值 |
|---|---|---|
| **detect**（YOLO11n，100 ep） | mAP@0.5 / 0.5:0.95 | **0.808** / **0.569** |
| | mAP@0.75 | 0.600 |
| | Precision / Recall | 0.787 / 0.755 |
| | FPS | **99.4**（10.1 ms/帧） |
| **pose**（YOLO11n-pose，30 ep） | 关键点(茎点) mAP@0.5 / 0.5:0.95 | **0.841** / **0.835** |
| | 框 mAP@0.5 / 0.5:0.95 | 0.768 / 0.524 |
| | Pose Precision | 0.788 |
| | FPS | **99.6**（10.0 ms/帧） |

**产物**：
- detect 权重：`runs/detect/runs/weed_yolo11n_full/weights/best.pt`
- pose 权重：`runs/pose/runs/weed_yolo11n_pose_full/weights/best.pt`
- detect 可视化：`demo/*.jpg`（GT 茎点版）；pose 可视化：`demo_pose/*.jpg`（模型预测茎点版，7 张）
- 方案设计报告：`report/design_report.pdf`（LaTeX/Tectonic 编译，含测试样例图）

**结论**：
1. detect 全量 mAP@0.5=0.808，较子集(0.758)提升约 5 点；实时性 99 FPS，超 30 FPS 目标 3 倍。
2. pose 关键点 mAP@0.5=0.841，但受「茎点≈框中心」标注质量限制，精度是「框中心」级，非真茎基部。
3. 部署链路（detect/pose 模型 → 茎短线段 → 作物 mask 膨胀避让 → 文本输出）端到端跑通。
