# VTS 4.0 — 视觉-触觉多模态传感器系统 全面参考手册

> **版本**: v4.0 | **更新日期**: 2026-05-15 | **运行环境**: Windows + Python 3.x + PyTorch

---

## 目录

1. [系统概述](#1-系统概述)
2. [硬件架构](#2-硬件架构)
3. [软件架构总览](#3-软件架构总览)
4. [核心原理](#4-核心原理)
5. [V0 数据采集准备](#5-v0-数据采集准备)
6. [V1 检测参数调优](#6-v1-检测参数调优)
7. [V2 首帧匹配与三维重建](#7-v2-首帧匹配与三维重建)
8. [V4 实时跟踪与方向感知](#8-v4-实时跟踪与方向感知)
9. [V5 多模态数据采集与编辑](#9-v5-多模态数据采集与编辑)
10. [V6 力预测模型训练与评估](#10-v6-力预测模型训练与评估)
11. [V7 实时力预测系统](#11-v7-实时力预测系统)
12. [压电传感器集成](#12-压电传感器集成)
13. [完整操作流程](#13-完整操作流程)
14. [数据格式规范](#14-数据格式规范)
15. [模型架构详解](#15-模型架构详解)
16. [文件清单](#16-文件清单)
17. [常见问题排查](#17-常见问题排查)

---

## 1. 系统概述

### 1.1 项目目标

VTS 4.0 是一套**视觉-压电多模态触觉感知系统**，通过以下三种传感器的融合，实现对接触力/力矩（6-DOF）的高精度估计：

| 传感器类型 | 频率 | 用途 |
|-----------|------|------|
| **双目视觉** (镜面反射式) | ~30 Hz | 60个标记点三维形变追踪 |
| **六维力传感器** (Robotiq FT) | ~100 Hz | 真实力/力矩标签 (训练监督信号) |
| **压电触觉传感器** (5×8通道) | ~kHz | 高频振动/压力动态信息 |

### 1.2 核心工作流

```
相机采集 → 标记点检测 → 3D重建 → dxyz位移计算 → 力预测
                ↑                          ↑
           压电传感器 ──→ 特征提取 ──→ 多模态融合
                ↑
           力传感器 ──→ 训练标签
```

### 1.3 关键指标

- **标记点数量**: 60个 (左右各约30个)
- **力预测输出**: 6维 (Fx, Fy, Fz, Mx, My, Mz)
- **算法推理速度**: ~1000+ FPS (纯推理)
- **系统总帧率**: ~30 FPS (受摄像头限制)
- **模型参数量**: ~277K (MLP基础版)

---

## 2. 硬件架构

### 2.1 整体结构

```
┌──────────────────────────────────────────────────┐
│                  工业相机 (USB)                     │
│                      ↓                            │
│              ┌──────────────┐                     │
│              │   镜面反射板   │                     │
│              │  (45° 双镜面) │                     │
│              └──────────────┘                     │
│              ↙              ↘                     │
│         左侧触觉面        右侧触觉面                │
│    (60个标记点分布)    (60个标记点分布)              │
│              ↓                ↓                   │
│         压电传感器        压电传感器                 │
│       (5组 × 8通道)     (5组 × 8通道)              │
└──────────────────────────────────────────────────┘
                      ↓
              六维力传感器 (底座)
```

### 2.2 镜面双目原理

使用**单相机 + 双镜面**构成立体视觉系统：
- 相机拍摄一张包含左右两个镜面反射图像的图片
- 图像沿垂直中轴分为左半部分和右半部分
- 左半部分对应左侧触觉面 (左侧镜面反射)
- 右半部分对应右侧触觉面 (右侧镜面反射)
- 通过双目立体标定获得左右"虚拟相机"的内外参数

### 2.3 压电传感器规格

| 参数 | 值 |
|------|-----|
| ADC组数 | 5组 (ADC1-ADC5) |
| 每组通道数 | 8通道 (CH1-CH8) |
| 采样精度 | 24位ADC (有效值23位) |
| 参考电压 | 3.3V |
| 通信接口 | 串口 (UART) |
| 波特率 | 921600 |
| 帧格式 | 帧头 0xAA 0xAA, 帧长 29字节, 帧尾 0xFF 0xFF |
| 通道特殊处理 | CH0/CH1 信号取反 |

### 2.4 六维力传感器 (Robotiq)

| 参数 | 值 |
|------|-----|
| 通信方式 | ZMQ Topic 订阅 (TCP) |
| 订阅地址 | `tcp://192.168.50.1:19091` |
| Topic | `system_rtstate` |
| 输出 | Fx, Fy, Fz (N), Mx, My, Mz (Nm) |
| 频率 | ~100 Hz |

---

## 3. 软件架构总览

### 3.1 文件版本演进

```
V0: 数据采集与标定准备
  ├── V0_MP4toPNG.py         视频录制 → PNG帧序列
  ├── V0_PNGtoMP4.py         PNG帧序列 → 视频
  ├── V0_ROI.py              ROI区域管理模块
  ├── V0_Vision-calibration.py 标定图像采集
  └── calibration.m          MATLAB标定参数导出

V1: 预处理
  └── V1_Preprocess.py       标记点检测参数调优

V2: 匹配重建
  └── V2_Match.py            手动点匹配 + 三维重建

V4: 实时跟踪
  └── V4_VpTac-direction.py  实时跟踪 + 方向分析

V5: 多模态采集
  ├── V5_Vptac-force.py      完整采集系统
  ├── V5_Vptac-simple_testOK.py 简化版采集
  └── V5_h5_edit.py          HDF5数据编辑器

V6: 模型训练/评估
  ├── V6_force_train.py          MLP训练
  ├── V6_force_train_pointnet.py PointNet训练
  ├── V6_force_train_lightnet.py LightNet训练
  ├── V6_force_predict.py        预测推理接口
  ├── V6_force_predict_pointnet.py
  ├── V6_force_predict_lightnet.py
  ├── V6_force_eval.py           评估
  ├── V6_force_eval_ablation.py  单架构消融
  └── V6_force_eval_cross_model.py 跨架构对比

V7: 实时预测
  ├── V7_Vptac-force-predict.py  实时预测系统
  └── V7_visualize_compare.py    可视化对比
```

### 3.2 依赖项

```
核心依赖:
  numpy, scipy, opencv-python, h5py, pyqtgraph
  PyQt5, matplotlib, torch, pyserial

机器人通信 (utils/):
  topic.pyd (ZMQ topic 订阅)
  message.py (消息解析)
  libprotobuf.dll, libzmq-v142-mt-4_3_6.dll
```

---

## 4. 核心原理

### 4.1 标记点检测算法 (CircleDetector)

所有版本共用同一检测器，核心采用**局部极小值检测法** (替代传统轮廓检测):

**步骤**:
1. **灰度化**: BGR → Gray
2. **CLAHE增强**: `clipLimit=2.0, tileGridSize=(8,8)` — 增强局部对比度，抵抗受压后亮度变化
3. **高斯滤波**: `GaussianBlur(blur, blur)`, blur限制 ≤7，防止密集点合并
4. **局部极小值检测**:
   - 对平滑图像做腐蚀 (窗口=平均半径×2+1)
   - `minima_mask = (smooth == eroded)` 找到局部极小值
   - 与自适应阈值二值化做 AND 过滤
   - 膨胀恢复形态
   - findContours 提取轮廓 → 计算质心
5. **KDTree去重**: 距离 < 2.5×平均半径的点对，保留前一个

**关键参数** (来自 `marker_params.json`):
```json
{
  "blur": 3,        // 高斯模糊核大小 (奇数, ≤7)
  "block_size": 11,  // 自适应阈值窗口
  "c_val": 5,        // 阈值补偿值
  "morph_size": 2,   // 形态学操作大小
  "min_area": 30,    // 最小轮廓面积
  "max_area": 1000,  // 最大轮廓面积
  "circularity": 70, // 圆度过滤 (%)
  "inertia": 30      // 形状规整度 (%)
}
```

### 4.2 三维重建原理

**线性三角测量** (Linear Triangulation):

给定标定参数 K1, D1, K2, D2, R, T:
1. `stereoRectify()` 计算校正矩阵 R1, R2, P1, P2 (带缓存，参数不变时复用)
2. 对左右匹配点分别 `undistortPoints()`
3. 对每对匹配点 (uL, vL) ↔ (uR, vR)，构建线性方程组:

```
A = [uL·P1[2,:] - P1[0,:]]
    [vL·P1[2,:] - P1[1,:]]
    [uR·P2[2,:] - P2[0,:]]
    [vR·P2[2,:] - P2[1,:]]
```

4. SVD 分解 A，取最小奇异值对应向量 V[-1]
5. 齐次化: `(X, Y, Z) = V[:3] / V[3]`

### 4.3 点跟踪算法

使用**匈牙利算法** (Hungarian Algorithm / linear_sum_assignment) 进行帧间点匹配:

1. 计算当前帧检测点与上一帧跟踪点的欧氏距离矩阵
2. 使用 `scipy.optimize.linear_sum_assignment` 最小化总匹配成本
3. 对未匹配的点标记为新增或丢失
4. 连续多帧异常则触发重新初始化

### 4.4 接触特征设计 (9维)

从60个标记点的位移向量 dxyz (60×3) 中提取:

| 索引 | 特征名 | 公式 | 物理意义 |
|------|--------|------|----------|
| 0 | contact_ratio | n_contact / 60 | 有效接触点比例 |
| 1 | disp_std | std(dxyz_norm) | 位移空间方差 |
| 2 | disp_max | max(dxyz_norm) | 最大位移 |
| 3 | disp_mean | mean(dxyz_norm) | 平均位移 |
| 4 | disp_sum | sum(dxyz_norm) | 总位移 |
| 5 | concentration | max / (mean + eps) | 集中度 |
| 6 | avg_contact_disp | sum / n_contact | 平均接触位移 |
| 7 | pressure_index | sum / (area × 0.01) | 压强指标 |
| 8 | entropy | -Σ(p × log(p)) | 位移分布熵 |

### 4.5 压电特征提取 (时间窗口统计法)

**核心问题**: 压电数据 kHz 级 vs 视觉数据 30Hz，频率不同。

**解决方案**: 对每个视觉帧时间戳 T，提取压电信号在 [T - window_ms, T] 窗口内的统计特征。

**5维特征**:

| 特征名 | 公式 | 物理意义 |
|--------|------|----------|
| mean | mean(vals) | 平均压力 |
| std | std(vals) | 压力波动 |
| rms | sqrt(mean(vals²)) | 有效值 |
| max | max(abs(vals)) | 峰值压力 |
| energy | sum(abs(vals)) | 累积能量 (绝对值积分) |

**默认时间窗口**: 33ms (对应30Hz视觉帧率)

---

## 5. V0 数据采集准备

### 5.1 V0_MP4toPNG.py — 视频录制

**功能**: 从USB摄像头实时采集并保存为PNG帧序列。

**启动**:
```bash
python V0_MP4toPNG.py
```

**操作流程**:
1. 自动连接摄像头 (索引2, 使用 DirectShow)
2. 预览画面实时显示
3. 点击"开始录制" → 自动创建时间戳命名的文件夹 (如 `Vptac_shape_data_20260512_200647`)
4. 帧以 `001.png, 002.png, ...` 命名保存
5. 点击"停止录制" → 弹出完成提示

**架构**:
- `CameraThread`: 生产者线程，60FPS采集 (MJPEG编码, 640×480)
- `ImageSaver`: 消费者线程，异步写盘避免IO阻塞
- `RecorderGUI`: PyQt5主界面

**摄像头配置**:
```python
CAMERA_INDEX = 2
FOURCC = 'MJPG'
RESOLUTION = 640 × 480
TARGET_FPS = 60
```

### 5.2 V0_PNGtoMP4.py — 帧序列转视频

**功能**: 将 PNG 帧序列合成为 MP4 视频文件。

**启动**:
```bash
python V0_PNGtoMP4.py
```

**参数说明**:
```python
image_folder = 'Vptac_shape_data_20260120_162028'  # 修改为实际目录
output_file = 'output_video.mp4'
fps = 30  # 输出帧率
```

**编码器**: `mp4v` (有损)，如需无损可改用 `FFV1` + `.avi` 格式。

### 5.3 V0_ROI.py — ROI区域管理

**功能**: 在图像上绘制多边形ROI区域，保存/加载掩膜，过滤检测点。

**作为独立工具使用**:
```bash
python V0_ROI.py
```
→ 弹出文件选择框 → 选择镜像双目图像 → 依次绘制左侧和右侧ROI → 自动保存为 `data/roi_masks.npz`

**作为模块导入**:
```python
from V0_ROI import auto_load_roi_masks, apply_roi_mask, create_roi_interactive

# 自动加载
left_mask, right_mask = auto_load_roi_masks(image_path)

# 应用过滤
valid_points = apply_roi_mask(detected_points, left_mask)
```

**操作说明**:
- 左键点击: 添加ROI顶点
- 中键点击: 撤销最后一个顶点
- 按 `q`: 完成当前ROI绘制 (至少需要3个点)

**保存格式**: `data/roi_masks.npz` 包含 `left_mask` 和 `right_mask` 两个数组。

### 5.4 V0_Vision-calibration.py — 标定图像采集

**功能**: 采集双目镜面标定图像 (棋盘格或圆点阵列)。

**启动**:
```bash
python V0_Vision-calibration.py
```

**操作流程**:
1. 摄像头实时预览
2. 将标定板放置在触觉面上，点击"采集并分割保存"
3. 图像自动沿中轴分割为左右两部分
4. 分别补齐为完整分辨率 (空白部分填充黑色)
5. 左图保存到 `calibration_left/`, 右图保存到 `calibration_right/`

**输出**:
```
calibration_left/
  ├── img_20260515_143025_123456.jpg
  └── ...
calibration_right/
  ├── img_20260515_143025_123456.jpg
  └── ...
```

### 5.5 calibration.m — MATLAB标定参数导出

**功能**: 将 MATLAB Stereo Camera Calibrator 的标定结果导出为 OpenCV 格式的 txt 文件。

**使用**:
1. 在 MATLAB 中运行 Stereo Camera Calibrator
2. 导出结果变量 `stereoParams` 到工作区
3. 运行 `calibration.m`

**输出** (保存到 `calibration/` 目录):
```
calibration/
  ├── K1.txt  (3×3 内参矩阵 - 左相机)
  ├── K2.txt  (3×3 内参矩阵 - 右相机)
  ├── D1.txt  (1×5 畸变系数 - 左相机)
  ├── D2.txt  (1×5 畸变系数 - 右相机)
  ├── R.txt   (3×3 旋转矩阵)
  └── T.txt   (3×1 平移向量)
```

**格式转换**:
- 内参矩阵: MATLAB存储为转置 → `K = IntrinsicMatrix'`
- 畸变系数: `[k1, k2, p1, p2]` → `[k1, k2, p1, p2, k3]`
- 旋转矩阵: `R_opencv = R_matlab'`
- 平移向量: `T_opencv = T_matlab'`

---

## 6. V1 检测参数调优

### 6.1 V1_Preprocess.py — 参数调整工具

**功能**: 加载样张图片，实时调整检测参数，可视化检测效果。

**启动**:
```bash
python V1_Preprocess.py
```

**操作流程**:
1. 点击"选择多张样张" → 选择最多12张图片
2. 自动加载该目录下的 `marker_params.json` (如果存在)
3. 自动加载 ROI 掩膜 (如果存在 `data/roi_masks.npz`)
4. 调整左侧8个滑块实时查看12个样张的检测效果
5. 点击"保存参数到样张目录" → 生成 `marker_params.json`
6. 可选: "开启批量处理" → 对整个文件夹应用当前参数

**可调参数**:
| 参数 | 范围 | 默认 | 说明 |
|------|------|------|------|
| blur | 1-15 | 3 | 高斯模糊核 (奇数) |
| block_size | 3-51 | 11 | 自适应阈值窗口 (奇数) |
| c_val | -10~20 | 5 | 阈值补偿 |
| morph_size | 0-10 | 2 | 形态学操作大小 |
| min_area | 5-500 | 30 | 最小轮廓面积 |
| max_area | 100-1000 | 1000 | 最大轮廓面积 |
| circularity | 0-100 | 70 | 圆度过滤百分比 |
| inertia | 0-100 | 30 | 形状规整度百分比 |

---

## 7. V2 首帧匹配与三维重建

### 7.1 V2_Match.py — 手动匹配与重建

**功能**: 对首帧图像进行左右标记点匹配，三维重建，保存匹配结果。

**启动**:
```bash
python V2_Match.py
```

**操作流程**:
1. 弹出文件选择框 → 选择图像文件
2. 如果不存在 ROI 掩膜，自动进入 ROI 绘制模式
3. 自动检测左右标记点
4. 进入手动匹配界面

**手动匹配操作**:
- 在左图点击或按住鼠标滑过圆点 → 选中并加入队列 (黄色，数字表示顺序)
- 在右图点击或按住鼠标滑过圆点 → 自动按顺序与队列首位配对
- 已匹配点显示为绿色
- 按 `z` 撤销 (优先撤销队列，再撤销已匹配对)
- 按 `c` 清空所有匹配
- 按 `s` 保存并退出
- 按 `ESC` 取消退出

**输出**:
```
data/
  ├── roi_masks.npz          # ROI掩膜
  ├── matched_points.npz     # 匹配结果
  └── output_points/         # 3D坐标txt
      └── reconstructed_points.txt

result/
  └── frame_000_points.txt   # V3格式: X Y Z Left_x Left_y Right_x Right_y
```

### 7.2 标定参数目录

用于存放 MATLAB 导出的标定参数，格式示例:
```
calibration_1mm_12X9_0512_2_Paras/
  ├── K1.txt, K2.txt
  ├── D1.txt, D2.txt
  ├── R.txt, T.txt
```

---

## 8. V4 实时跟踪与方向感知

### 8.1 V4_VpTac-direction.py

**功能**: 实时视频/摄像头标记点跟踪、三维重建、压电传感器波形显示。

**启动**:
```bash
python V4_VpTac-direction.py
```

**启动配置** (弹窗依次选择):
1. 标定参数文件夹 (如 `calibration_1mm_12X9_0512_2_Paras/`)
2. 参数 JSON 文件 (`marker_params.json`)
3. 首帧 ROI 和匹配数据保存文件夹
4. 选择输入源: 视频文件 / USB 摄像头

**GUI 功能**:
- 左侧: 视频帧预览 (带检测标记)
- 右侧: Matplotlib 3D点云实时显示
- 支持暂停/播放、手动旋转3D视角
- "设置为首帧"按钮: 动态更新基准帧
- 压电传感器串口连接和波形显示

**核心线程**:
- `CameraThread`: 摄像头采集 (60FPS)
- `FrameProcessThread`: 后台帧处理 (检测+重建)
- `SerialReceiver`: 压电串口数据接收

**3D视角控制**: 鼠标拖拽旋转，`view_elev=90, view_azim=0, view_roll=-90` 为默认俯视视角。

---

## 9. V5 多模态数据采集与编辑

### 9.1 V5_Vptac-force.py — 完整采集系统

**功能**: 同步采集视觉(30Hz) + 力传感器(100Hz) + 压电传感器(kHz)，保存为统一HDF5文件。

**启动**:
```bash
python V5_Vptac-force.py
```

**启动配置** (弹窗依次选择):
1. 标定参数文件夹
2. 参数 JSON 文件
3. 数据保存文件夹
4. 力预测模型目录 (可选)
5. 选择输入源: 视频文件 / USB 摄像头

**GUI 面板**:

| 区域 | 控件 | 说明 |
|------|------|------|
| **点云** | 3D显示 | Matplotlib 实时点云 |
| **力传感器** | 6通道图表 | PyQtGraph 实时力曲线 |
| **压电传感器** | 波形 + 串口选择 | PyQtGraph 压电波形 |
| **控制** | 开始采集/停止 | 含采集计时和帧计数 |
| **压电控制** | ADC组选择 | ADC1-ADC5 下拉框 |
| | 通道选择 | CH1-CH8 下拉框 |

**HDF5 存储结构**:
```
calibration_YYYYMMDD_HHMMSS.h5
├── /vision
│   ├── timestamp       (T,)      float64  视觉时间戳
│   ├── xyz             (T, N, 3) float32  当前3D坐标
│   ├── dxyz            (T, N, 3) float32  位移 (当前 - 基准)
│   └── abnormal        (T,)      bool     异常帧标记
├── /piezo              (保存时选中的ADC组)
│   ├── timestamp       (M,)      float64  压电采样时间戳
│   ├── values          (M, 8)    float32  8通道电压值
│   ├── @n_channels     8
│   ├── @adc_group      1-5       (保存时选中的ADC组)
│   └── @unit           "V"
├── /piezo_stream       (流式录制，所有ADC组)
│   ├── /adc1
│   │   ├── timestamp   (N1,)     float64
│   │   └── values      (N1, 8)   float32
│   ├── /adc2 ...
│   └── /adc5 ...
└── /reference
    └── xyz_ref         (N, 3)    float32  基准3D坐标
```

**力数据**保存在单独的 `ft_calibration_YYYYMMDD_HHMMSS.h5`:
```
/ft
  ├── timestamp  (K,)   float64
  └── values     (K, 6) float32  [fx,fy,fz,mx,my,mz]
```

**流式录制机制**:
- 采集期间压电数据直接追加写入 HDF5 (`piezo_stream` 组)
- 每200帧刷新一次到磁盘
- 避免长时间采集导致内存溢出
- 可采集全部5个ADC组的所有8通道数据

**关键代码位置**:
- 压电采集线程: `V5_Vptac-force.py:50-305` (`PiezoSerialThread`)
- 主采集逻辑: `V5_Vptac-force.py` 的 `V5MainWindow` 类
- 力传感器订阅: ZMQ topic 回调

### 9.2 V5_Vptac-simple_testOK.py — 简化版

**功能**: 去除力传感器和压电传感器，仅保留视觉跟踪和3D显示。

适用场景: 调试视觉算法、快速验证。

### 9.3 V5_h5_edit.py — HDF5 数据编辑器

**功能**: 对采集的HDF5数据进行时间对齐、有效段筛选、压电特征提取和保存。

**启动**:
```bash
python V5_h5_edit.py
```

**操作流程**:

1. **打开文件**
   - "打开点云文件" → 选择 `calibration_*.h5`
   - "打开力觉文件" → 选择 `ft_calibration_*.h5`
   - 或 "打开处理文件" → 加载已处理的 `processed_*.h5` 再编辑

2. **时间对齐**
   - 通过滑块调整时间偏移 (offset)
   - 观察力传感器曲线与视觉帧的对齐情况
   - 视觉帧时间戳作为基准，力数据通过最近邻匹配对齐

3. **有效段筛选**
   - 在力传感器图上拖动鼠标框选有效数据段
   - 每个段用彩色区间标记
   - 可添加/删除段

4. **压电传感器设置** (如果存在压电数据)
   - 通道选择: CH1-CH8
   - ADC组选择: ADC1-ADC5
   - 时间窗口: 10-100ms (默认33ms)
   - 50Hz陷波滤波: 开关
   - 尖峰滤除: 开关
   - 实时波形预览窗口

5. **保存处理数据**
   - 点击"保存处理数据" → 输出 `processed_YYYYMMDD_HHMMSS.h5`

**输出 HDF5 结构**:
```
processed_YYYYMMDD_HHMMSS.h5
├── /vision
│   ├── timestamp     (T,)      float64
│   ├── dxyz          (T, 60, 3) float32
│   └── predicted_force (T, 6)  float32 (如果有)
├── /force
│   ├── timestamp     (T,)      float64
│   └── values        (T, 6)    float32  [fx,fy,fz,mx,my,mz]
├── /piezo            (如果启用)
│   ├── features      (T, 5)    float32  [mean,std,rms,max,energy]
│   ├── @feature_names "mean,std,rms,max,energy"
│   ├── @window_ms    33
│   └── @source_channel 通道号
├── /segments
│   ├── @count        N
│   ├── @seg_0_start  (相对时间)
│   └── @seg_0_end
└── /reference
    └── xyz_ref       (60, 3)   float32
```

**压电50Hz陷波滤波器**:
- 使用 `scipy.signal.iirnotch` 设计 IIR 陷波滤波器
- 品质因数 Q=30
- 在提取时间窗口特征之前应用

**关键代码位置**:
- 数据加载: `V5_h5_edit.py:load_force_file()`
- 时间对齐: `V5_h5_edit.py:_update_time_offset()`
- Segment管理: `V5_h5_edit.py:_on_mouse_press/release()`
- 压电特征提取: 核心函数 `extract_piezo_window_features()`
  ```python
  def extract_piezo_window_features(vision_timestamp, piezo_ts, piezo_vals, window_ms=33):
      t_start = vision_timestamp - window_ms / 1000.0
      mask = (piezo_ts >= t_start) & (piezo_ts <= vision_timestamp)
      window_vals = piezo_vals[mask]
      return {
          'mean': np.mean(window_vals),
          'std': np.std(window_vals),
          'rms': np.sqrt(np.mean(window_vals**2)),
          'max': np.max(np.abs(window_vals)),
          'energy': np.sum(np.abs(window_vals))
      }
  ```
- 保存逻辑: `V5_h5_edit.py:_save_processed()`

---

## 10. V6 力预测模型训练与评估

### 10.1 模型总览

| 模型 | 训练脚本 | 架构特点 | 参数量 |
|------|---------|---------|--------|
| **MLP** (BaseNet) | `V6_force_train.py` | 展平+接触特征MLP | ~277K |
| **PointNet** | `V6_force_train_pointnet.py` | PointNet点云特征学习 | 可变 |
| **LightNet** | `V6_force_train_lightnet.py` | 轻量级卷积网络 | 可变 |

### 10.2 V6_force_train.py — MLP 训练

**启动**:
```bash
# 基础训练 (仅视觉)
python V6_force_train.py --data_dir force_calibration --epochs 500 --batch_size 256 --lr 5e-4

# 多模态训练 (视觉+压电)
python V6_force_train.py --data_dir force_calibration --use_piezo --epochs 500 --batch_size 256 --lr 5e-4
```

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `force_calibration` | 数据目录 (含 processed_*.h5) |
| `--output_dir` | 数据目录下自动创建 | 模型输出目录 |
| `--epochs` | 500 | 训练轮次 |
| `--batch_size` | 256 | 批大小 |
| `--lr` | 5e-4 | 学习率 |
| `--use_piezo` | False | 是否使用压电特征 |
| `--noise_std` | 0.01 | 训练噪声增强标准差 |
| `--contact_threshold` | 0.02 | 接触判定阈值 |

**输入维度**:
- Vision-only: 180 (dxyz展平) + 9 (接触特征) = **189维**
- Vision+Piezo: 180 + 9 + 5 (压电特征) = **194维**

**ForceMLP 网络结构**:
```
输入 (189/194) → Point分支 (180) + Contact分支 (9/14)
                        ↓                    ↓
                   直接传入          ContactNet (9→32→32)
                        ↓                    ↓
                        └──── 拼接 (180+32) ──┘
                                  ↓
                    FC(512) → BN → ReLU → Drop(0.2)
                    FC(256) → BN → ReLU → Drop(0.1)
                    FC(128) → BN → ReLU
                    FC(6)   → 输出 [fx,fy,fz,mx,my,mz]
```

**数据处理流程**:

1. **文件选择**: 交互式选择训练/验证文件和测试文件
2. **Segment划分**: 按连续段划分训练/验证集 (避免相邻帧泄漏)
3. **Scaler计算**: 仅基于训练集计算标准化参数
4. **训练循环**:
   - 损失函数: MSE Loss
   - 优化器: AdamW (weight_decay=1e-5)
   - 学习率调度: ReduceLROnPlateau (patience=20, factor=0.5) + CosineAnnealingLR
   - 早停: patience=50
   - 数据增强: 高斯噪声 (std=0.01)

**输出**:
```
model_output/
├── model.pth            # 模型权重
├── scaler.npz           # 标准化参数
├── train_config.json    # 训练配置
├── split_indices.npz    # 数据集划分索引
├── train_history.npz    # 训练曲线数据
└── evaluation/          # 评估结果 (由 V6_force_eval.py 生成)
    ├── metrics.json
    ├── scatter.png
    ├── timeseries.png
    ├── error_dist.png
    ├── train_history.png
    └── ...
```

### 10.3 V6_force_predict.py — 推理接口

**作为模块使用**:
```python
from V6_force_predict import ForcePredictor

# 加载模型
predictor = ForcePredictor("force_calibration/model_output")

# 单帧推理
dxyz = np.array(...)  # shape (60, 3) — 60个标记点位移
force = predictor.predict(dxyz)  # shape (6,) — [fx,fy,fz,mx,my,mz]

# 带压电特征的推理
piezo_feat = np.array([mean, std, rms, max, energy])  # shape (5,)
force = predictor.predict(dxyz, piezo_feat=piezo_feat)

# 批量推理
dxyz_batch = np.array(...)  # shape (100, 60, 3)
forces = predictor.predict_batch(dxyz_batch)  # shape (100, 6)
```

**类结构**:
```python
class ForcePredictor:
    def __init__(self, model_dir, contact_threshold=0.02)
        # 加载 train_config.json, scaler.npz, model.pth
        # 预计算 GPU tensor 版 scaler 加速推理

    def predict(self, dxyz, piezo_feat=None) -> np.ndarray
        # dxyz(60,3) → flatten(180) → +contact_feat(9) → +piezo_feat(5?)
        # → normalize → model → denormalize → (6,)

    def predict_batch(self, dxyz_batch, piezo_feat_batch=None) -> np.ndarray
        # 批量推理，同上但 (T, ...) → (T, 6)
```

**独立测试**:
```bash
python V6_force_predict.py --model_dir force_calibration/model_output
```

### 10.4 V6_force_eval.py — 模型评估

**启动**:
```bash
# 交互式 (自动弹出模型目录选择框)
python V6_force_eval.py

# 命令行
python V6_force_eval.py --model_dir force_calibration/model_output --data_dir force_calibration
```

**评估指标**:
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **R²**: 决定系数
- **MaxError**: 最大绝对误差
- **MAPE%**: 平均绝对百分比误差 (排除|真实值|<0.1的样本)

**生成图表** (保存到 `evaluation/`):
| 图表 | 说明 |
|------|------|
| `scatter.png` | 预测 vs 真实 散点图 (6个子图) |
| `timeseries.png` | 时间序列对比 (6行) |
| `error_dist.png` | 误差分布直方图 |
| `error_vs_magnitude.png` | 误差 vs 力大小 |
| `contact_analysis.png` | 平面 vs 尖锐物体预测差异 |
| `train_history.png` | 训练/验证 loss 曲线 |

### 10.5 V6_force_eval_ablation.py — 消融实验

**功能**: 对比 Vision-only 和 Vision+Piezo 两个模型在同一测试集上的表现。

**启动**:
```bash
python V6_force_eval_ablation.py \
  --baseline_dir force_calibration/model_output_vision_only \
  --multimodal_dir force_calibration/model_output_vision_piezo
```

**输出** (保存到 `ablation_comparison/`):
```
ablation_comparison/
├── ablation_scatter.png      # 散点对比 (蓝色=Vision-only, 红色=Vision+Piezo)
├── ablation_timeseries.png   # 时序对比 (黑=真实, 蓝=Vision-only, 红=Vision+Piezo)
├── ablation_metrics.png      # 指标柱状图 (MAE/RMSE/R² 对比)
└── ablation_metrics.json     # 数值结果
```

**打印格式**:
```
分量   指标   Vision-only   Vision+Piezo   改进
fx     MAE      0.1234        0.0987      +20.02%
fx     RMSE     0.2345        0.1876      +20.00%
fx     R2       0.9123        0.9456      +0.0333
...
```

### 10.6 V6_force_eval_cross_model.py — 跨架构对比

**功能**: 同时对比 MLP / PointNet / LightNet 三种架构的 Vision-only 和 Vision+Piezo 模型。

**启动**:
```bash
python V6_force_eval_cross_model.py
```

**输出**: 综合对比表和图表 (保存到 `cross_model_comparison/`)。

### 10.7 其他训练脚本

**PointNet 训练**:
```bash
python V6_force_train_pointnet.py --data_dir force_calibration --use_piezo --epochs 500
```

**LightNet 训练**:
```bash
python V6_force_train_lightnet.py --data_dir force_calibration --use_piezo --epochs 500
```

**对应的预测接口**:
- `V6_force_predict_pointnet.py` — PointNet 推理
- `V6_force_predict_lightnet.py` — LightNet 推理

**对应的消融脚本**:
- `V6_force_eval_ablation_pointnet.py`
- `V6_force_eval_ablation_lightnet.py`

---

## 11. V7 实时力预测系统

### 11.1 V7_Vptac-force-predict.py — 实时预测

**功能**: 实时视频采集 → 标记点检测 → 3D重建 → 力预测 → 可视化 + HDF5记录。

**启动**:
```bash
python V7_Vptac-force-predict.py
```

**启动配置** (弹窗依次选择):
1. 标定参数文件夹
2. 参数 JSON 文件 (`marker_params.json`)
3. 数据保存文件夹
4. 力预测模型目录 (如 `force_calibration/model_output`)
5. 选择输入源: 视频文件 / USB 摄像头

**GUI 面板**:

| 区域 | 控件 | 说明 |
|------|------|------|
| **视频** | 视频帧预览 | 带检测标记点 |
| **点云** | Matplotlib 3D显示 | 实时重建+预测力箭头 |
| **力预测/实际** | 6通道曲线 | PyQtGraph 预测(红) vs 实际(蓝) |
| **压电波形** | 单通道波形 | 当前选中ADC组+通道 |
| **控制** | 开始采集/停止 | 统一切换 |
| **压电控制** | ADC组 + 通道 | 下拉框选择 |
| **HDF5回放** | 加载/播放/滑块 | 回放已保存的数据 |
| **调零** | 自动/手动 | 力传感器偏置清零 |

**数据记录** (统一 HDF5):
```
recordings/V7_recording_YYYYMMDD_HHMMSS.h5
├── /vision (30Hz)
│   ├── timestamp       (T,)      float64
│   ├── xyz             (T, 60, 3) float32
│   ├── dxyz            (T, 60, 3) float32
│   ├── abnormal        (T,)      bool
│   └── predicted_force (T, 6)    float32
├── /force (100Hz)
│   ├── timestamp       (K,)      float64
│   └── values          (K, 6)    float32
└── /reference
    └── xyz_ref         (60, 3)   float32
```

**实时预测流程** (`_predict_force` 方法):
```
1. 获取当前帧检测点 → 计算3D坐标
2. 计算 dxyz = 当前3D坐标 - 基准3D坐标 (60×3)
3. 如果模型需要压电:
   a. 从压电线程获取当前选中ADC组+通道的时间窗口数据
   b. 计算5维统计特征 [mean, std, rms, max, energy]
4. 调用 ForcePredictor.predict(dxyz, piezo_feat)
5. 返回 6维力预测值
```

**压电实时特征提取**:
```python
def extract_realtime_piezo_features(current_timestamp, window_ms=33):
    t_start = current_timestamp - window_ms / 1000.0
    window_vals = piezo_thread.get_window_values(t_start, current_timestamp)
    if len(window_vals) < 2:
        return np.zeros(5)
    vals = np.array(window_vals)
    return np.array([
        np.mean(vals), np.std(vals),
        np.sqrt(np.mean(vals**2)), np.max(np.abs(vals)),
        np.sum(np.abs(vals))
    ])
```

**力传感器调零**:
- 自动: 采集最初100帧的平均值作为偏置 (bias)
- 手动: 点击"调零"按钮 → 取最近100个采样的平均值

**HDF5回放**:
- 加载 `V7_recording_*.h5` 文件
- 力传感器数据通过最近邻时间戳匹配同步到视觉帧
- 播放速度 ~30FPS

### 11.2 V7_visualize_compare.py — 可视化对比

**功能**: 加载 V7 录制的 HDF5 文件，对比显示预测力与实际力。

**启动**:
```bash
python V7_visualize_compare.py
```

**GUI 布局**:
- 上方: 6个横向坐标轴 (Fx, Fy, Fz, Mx, My, Mz) — 预测 vs 实际
- 下方: Matplotlib 3D点云 + 预测力箭头

---

## 12. 压电传感器集成

### 12.1 独立工具

#### Tactile_Finger.py — 压电数据采集GUI

**功能**: 独立的压电传感器多通道波形显示工具。

**启动**:
```bash
python Tactile_Finger.py
```

**功能**:
- 8通道实时波形显示 (PyQtGraph, ±3.3V量程)
- 5个ADC组切换
- 串口自动扫描
- 200ms刷新周期

#### analyze_piezo_spectrum.py — 频谱分析

**功能**: 分析压电信号的频谱特性，识别噪声主频 (特别是50Hz工频干扰)。

**启动**:
```bash
# 命令行
python analyze_piezo_spectrum.py <h5文件路径> <ADC组> <通道> [最大频率]

# 交互式
python analyze_piezo_spectrum.py
```

**分析内容**:
1. 时域波形
2. 功率谱密度 (PSD, Welch方法)
3. 低频段放大 (0-200Hz)
4. 前10个主频峰 (标记50/100/150Hz工频)

**典型发现**: 50Hz 及其倍频 (100Hz, 150Hz) 是主要干扰源。

### 12.2 压电数据帧格式

```
帧结构 (29 字节):
┌────────┬────────┬──────────────────────┬────────┐
│ 0xAA   │ 0xAA   │  Payload (25 字节)    │ 0xFF   │ 0xFF   │
│ 帧头    │ 帧头    │                      │ 帧尾    │ 帧尾    │
└────────┴────────┴──────────────────────┴────────┴────────┘

Payload 结构 (25 字节):
┌──────────┬──────────────────────────────────┐
│ ADC组ID  │  8通道 × 3字节 (24位有符号整数)    │
│ (1字节)  │  CH0 CH1 CH2 CH3 CH4 CH5 CH6 CH7  │
└──────────┴──────────────────────────────────┘

字节解码:
  byte1 << 16 | byte2 << 8 | byte3
  若 bit23=1 (负数): 减去 0x1000000 (符号扩展)
  电压 = 解码值 / 2^23 × 3.3V
  CH0, CH1 取反
```

### 12.3 集成架构

```
V5采集阶段:
  PiezoSerialThread → _group_bufs[0..4] → 流式写入 piezo_stream
                                          → 停止时快照写入 piezo

V5_h5_edit阶段:
  读取 piezo 原始数据 → 应用滤波器 → 时间窗口特征提取 → 保存 piezo/features

V6训练阶段:
  读取 processed_*.h5 → 拼接 piezo/features → 194维输入训练

V7预测阶段:
  PiezoSerialThread → _group_bufs → 实时窗口特征提取 → 拼接预测输入
```

### 12.4 消融实验结果 (参考)

**Vision+Piezo vs Vision-only** (MLP, 加载过程, 50Hz滤波+尖峰滤除):

| 分量 | MAE 改进 | R² 改进 |
|------|---------|--------|
| fx | -9.49% | +0.0028 |
| fy | -5.78% | +0.0089 |
| fz | -9.49% | +0.0019 |
| mx | -17.50% | +0.0192 |
| my | -11.16% | +0.0062 |
| mz | -11.75% | +0.0132 |

---

## 13. 完整操作流程

### 13.1 初次设置流程

```
第一步: 硬件标定
  1. 使用标定板拍摄多组双目图像 → V0_Vision-calibration.py
  2. MATLAB Stereo Camera Calibrator 标定
  3. 运行 calibration.m 导出参数到 calibration/ 目录
  4. 复制标定参数文件夹，如 calibration_1mm_12X9_XXXX_Paras/

第二步: ROI设置
  1. 拍摄一张触觉面图像 → V0_MP4toPNG.py
  2. V0_ROI.py 绘制左右ROI → data/roi_masks.npz

第三步: 参数调优
  1. V1_Preprocess.py 加载12张样张
  2. 调整参数至检测效果满意
  3. 保存 marker_params.json

第四步: 首帧匹配
  1. V2_Match.py 加载图像
  2. 手动匹配左右标记点
  3. 保存匹配结果

第五步: 验证跟踪
  1. V4_VpTac-direction.py 验证实时跟踪效果
  2. 确认标记点跟踪稳定
```

### 13.2 数据采集流程

```
1. 启动 V5_Vptac-force.py
2. 依次选择: 标定文件夹 → 参数JSON → 保存目录 → 模型目录 → 输入源
3. 连接压电传感器串口
4. 选择压电 ADC组和通道
5. 设置力传感器调零偏置
6. 点击"开始采集"
7. 操作触觉面进行按压
8. 点击"停止采集" → 自动保存 HDF5

输出:
  calibration_YYYYMMDD_HHMMSS.h5  (视觉 + 压电原始 + 预测)
  ft_calibration_YYYYMMDD_HHMMSS.h5  (力传感器)
```

### 13.3 数据筛选与对齐流程

```
1. 启动 V5_h5_edit.py
2. 打开点云文件 → calibration_*.h5
3. 打开力觉文件 → ft_calibration_*.h5
4. 调整时间偏移 (如需微调视觉-力对齐)
5. 在力传感器图上框选有效按压段
6. 设置压电参数: 通道, ADC组, 时间窗口, 滤波器
7. 保存处理数据 → processed_*.h5
```

### 13.4 模型训练流程

```bash
# 1. 训练 Vision-only 基线
python V6_force_train.py --data_dir force_calibration --epochs 500 --batch_size 256

# 2. 训练 Vision+Piezo 多模态
python V6_force_train.py --data_dir force_calibration --use_piezo --epochs 500

# 3. 评估
python V6_force_eval.py --model_dir force_calibration/model_output

# 4. 消融对比
python V6_force_eval_ablation.py \
  --baseline_dir force_calibration/model_output_vision_only \
  --multimodal_dir force_calibration/model_output_vision_piezo
```

### 13.5 实时预测流程

```
1. 启动 V7_Vptac-force-predict.py
2. 依次选择: 标定文件夹 → 参数JSON → 保存目录 → 模型目录 → 输入源
3. 连接压电串口 (如果模型需要压电)
4. 选择压电 ADC组和通道
5. 点击"开始采集"
6. 观察实时力预测 vs 实际力传感器值
7. 停止后自动保存 HDF5 记录
```

---

## 14. 数据格式规范

### 14.1 原始采集文件 (calibration_*.h5)

```
/vision
  timestamp:    (T,)     float64   视觉帧时间戳 (秒)
  xyz:          (T,N,3)  float32   当前3D坐标 (mm)
  dxyz:         (T,N,3)  float32   位移 (mm)
  abnormal:     (T,)     bool      异常帧标记
  predicted_force: (T,6) float32   模型预测力 (若加载了模型)

/piezo
  timestamp:    (M,)     float64   压电采样时间戳
  values:       (M,8)    float32   8通道电压值 (V)
  @n_channels:  8
  @adc_group:   1-5
  @unit:        "V"

/piezo_stream  (如果启用流式录制)
  /adc1/timestamp:  (N1,)  float64
  /adc1/values:     (N1,8) float32
  /adc2/...
  /adc5/...

/reference
  xyz_ref:      (N,3)    float32   基准3D坐标
```

### 14.2 力传感器原始文件 (ft_calibration_*.h5)

```
/ft
  timestamp:    (K,)     float64   力传感器时间戳
  values:       (K,6)    float32   [fx,fy,fz,mx,my,mz]
```

### 14.3 处理后文件 (processed_*.h5)

```
/vision
  timestamp:    (T,)     float64
  dxyz:         (T,60,3) float32   位移

/force
  timestamp:    (T,)     float64   对齐后的时间戳
  values:       (T,6)    float32   对齐后的力值

/piezo
  features:     (T,5)    float32   [mean,std,rms,max,energy]
  @feature_names: "mean,std,rms,max,energy"
  @window_ms:   33
  @source_channel: 4

/segments
  @count:       N
  @seg_0_start: 相对时间
  @seg_0_end:   相对时间

/reference
  xyz_ref:      (60,3)   float32
```

### 14.4 V7 录制文件 (V7_recording_*.h5)

```
/vision (30Hz)
  timestamp:       (T,)      float64
  xyz:             (T,60,3)  float32
  dxyz:            (T,60,3)  float32
  abnormal:        (T,)      bool
  predicted_force: (T,6)     float32

/force (100Hz)
  timestamp:       (K,)      float64
  values:          (K,6)     float32

/reference
  xyz_ref:         (60,3)    float32
```

### 14.5 模型文件 (model_output/)

```
model_output/
├── model.pth            PyTorch 模型权重
├── scaler.npz           标准化参数 {x_mean, x_std, y_mean, y_std, bias}
├── train_config.json    训练配置
│   {
│     "input_dim": 189,
│     "output_dim": 6,
│     "hidden_dims": [512, 256, 128],
│     "dropout": [0.2, 0.1, 0.0],
│     "use_piezo": false,
│     "contact_threshold": 0.02,
│     "train_val_files": [...],
│     "test_files": [...]
│   }
├── split_indices.npz    数据划分 {train, val, test}
├── train_history.npz    训练历史 {train_loss, val_loss}
└── evaluation/
    ├── metrics.json
    └── *.png
```

### 14.6 ROI 文件

```
data/roi_masks.npz
  left_mask:   (H, W) uint8   255=ROI内, 0=ROI外
  right_mask:  (H, W) uint8
```

### 14.7 匹配文件

```
data/matched_points.npz
  left_points:   (N, 3)    左图检测点 [(x,y,r), ...]
  right_points:  (M, 3)    右图检测点 [(x,y,r), ...]
  matched_pairs: (K, 2)    匹配索引对 [(i_left, j_right), ...]
  mirror_axis:   int       图像分割中轴
  image_shape:   [H, W]    图像尺寸
```

---

## 15. 模型架构详解

### 15.1 MLP (BaseNet)

```
输入层: 189维 (180 dxyz + 9 contact) 或 194维 (+5 piezo)
├── ContactNet (接触特征子网络)
│   ├── Linear(9/14 → 32) + ReLU
│   └── Linear(32 → 32) + ReLU
│
├── 主干网络
│   ├── Linear(212 → 512) + BatchNorm + ReLU + Dropout(0.2)
│   ├── Linear(512 → 256) + BatchNorm + ReLU + Dropout(0.1)
│   ├── Linear(256 → 128) + BatchNorm + ReLU
│   └── Linear(128 → 6)
│
输出: [fx, fy, fz, mx, my, mz]
```

**设计要点**:
- dxyz(180)直接传入不经过子网络，保留原始空间信息
- 接触特征(9)和压电特征(5)通过 ContactNet 编码为32维
- BatchNorm 加速收敛
- Dropout 防止过拟合 (eval时自动关闭)

### 15.2 PointNet

```
输入: 点云 P (B, 60, 3) + 可选压电特征 P_piezo (B, 5)
├── Input Transform (T-Net 3×3): 学习点云旋转不变性
├── MLP(3→64→128→1024) + MaxPool → 全局特征 (B, 1024)
├── Feature Transform (T-Net 64×64): 特征空间对齐
├── 压电特征处理: Linear(5→128) + 拼接 (如果启用)
├── FC(1024+128 → 512 → 256 → 6)
输出: [fx, fy, fz, mx, my, mz]
```

### 15.3 LightNet

```
输入: 点云 dxyz (B, 60, 3) + 可选压电特征 P_piezo (B, 5)
├── Conv1D(3→32) + Conv1D(32→64) + Conv1D(64→128)
├── GlobalMaxPool → 全局特征 (B, 128)
├── 压电特征: Linear(5→64) + 拼接 (如果启用)
├── FC(128+64 → 128 → 64 → 6)
输出: [fx, fy, fz, mx, my, mz]
```

**特点**: 使用1D卷积替代PointNet的MLP，参数量更少，训练更快。

---

## 16. 文件清单

### 16.1 核心功能文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `Tactile_Finger.py` | 269 | 压电传感器独立采集GUI |
| `V0_MP4toPNG.py` | 268 | 摄像头录制 → PNG序列 |
| `V0_PNGtoMP4.py` | 71 | PNG序列 → 视频 |
| `V0_ROI.py` | 248 | ROI区域管理模块 |
| `V0_Vision-calibration.py` | 197 | 标定图像采集 |
| `V1_Preprocess.py` | 399 | 检测参数调优GUI |
| `V2_Match.py` | 399 | 手动匹配 + 3D重建 |
| `V4_VpTac-direction.py` | ~2400 | 实时跟踪 + 压电 |
| `V5_Vptac-force.py` | ~2600 | 完整多模态采集系统 |
| `V5_Vptac-simple_testOK.py` | ~1800 | 简化版采集 |
| `V5_h5_edit.py` | ~2100 | HDF5数据编辑器 |
| `V6_force_train.py` | ~800 | MLP训练 |
| `V6_force_train_pointnet.py` | ~950 | PointNet训练 |
| `V6_force_train_lightnet.py` | ~850 | LightNet训练 |
| `V6_force_predict.py` | 256 | MLP推理接口 |
| `V6_force_predict_pointnet.py` | ~300 | PointNet推理接口 |
| `V6_force_predict_lightnet.py` | ~300 | LightNet推理接口 |
| `V6_force_eval.py` | 368 | 模型评估 |
| `V6_force_eval_ablation.py` | 320 | 消融对比 |
| `V6_force_eval_cross_model.py` | ~700 | 跨架构对比 |
| `V7_Vptac-force-predict.py` | ~2700 | 实时力预测系统 |
| `V7_visualize_compare.py` | ~250 | HDF5可视化对比 |
| `analyze_piezo_spectrum.py` | 146 | 压电频谱分析 |
| `compare_models.py` | 83 | 多模型指标对比图 |
| `calibration.m` | 78 | MATLAB标定导出 |

### 16.2 支持文件

| 文件/目录 | 说明 |
|----------|------|
| `utils/main.py` | ZMQ topic 订阅示例 (力传感器) |
| `utils/message.py` | 力传感器消息解析 |
| `utils/topic.pyd` | ZMQ topic 通信模块 |
| `utils/libprotobuf.dll` | Protobuf 运行时 |
| `utils/libzmq-v142-mt-4_3_6.dll` | ZMQ 运行时 |
| `pressure_sensor/` | (预留) 压力传感器相关 |
| `recordings/` | V7录制文件输出目录 |
| `force_calibration/` | 力标定数据和模型 |
| `force_calibration_for0/` | (前期) 力标定数据 |
| `calibration*/` | 标定参数目录 (多组) |

### 16.3 文档文件

| 文件 | 说明 |
|------|------|
| `压电传感器集成修改总结.md` | 压电集成开发总结 |
| `训练记录.md` | 训练实验记录和指标对比 |
| `VTS4.0-全面参考手册.md` | 本手册 |

---

## 17. 常见问题排查

### 17.1 摄像头问题

**问题**: 摄像头无法打开或黑屏

**排查**:
1. 检查 `CAMERA_INDEX` 是否正确 (默认2)
2. 确认摄像头未被其他程序占用
3. 检查 USB 连接是否稳定
4. 尝试修改 FOURCC 编码 (MJPG → D3D11 或 YUY2)
5. 确认 `cv2.CAP_DSHOW` 是否必要 (Linux环境需移除)

### 17.2 标记点检测问题

**问题**: 检测点数量不对或位置偏移

**排查**:
1. 调整 `marker_params.json` 中的参数 (尤其是 blur 和 c_val)
2. 检查 ROI 掩膜是否正确覆盖检测区域
3. 检查光照条件是否均匀 (CLAHE 有一定补偿能力)
4. 确认图像没有被意外旋转 (V0中已注释掉180°旋转)

### 17.3 3D重建问题

**问题**: 点云形状异常或深度不准

**排查**:
1. 确认标定参数 (K1, D1, K2, D2, R, T) 正确
2. 检查首帧匹配是否正确
3. 确认 mirror_axis 正确分割左右图像
4. 检查 `stereoRectify` 的 alpha 参数 (默认0.9)

### 17.4 压电传感器问题

**问题**: 压电波形显示为空

**排查**:
1. 确认串口号正确 (扫描自动检测)
2. 确认波特率 = 921600
3. 确认串口未被其他程序占用
4. 用示波器验证传感器是否有信号输出
5. 检查帧格式: 帧头 0xAA 0xAA, 帧尾 0xFF 0xFF

**问题**: 压电信号噪声大

**解决方案**:
1. 启用50Hz陷波滤波器 (V5_h5_edit.py 中)
2. 启用尖峰滤除
3. 使用 `analyze_piezo_spectrum.py` 分析频谱
4. 建议选择合适的通道和ADC组 (通过V5实时波形观察)

### 17.5 力传感器问题

**问题**: 力传感器无数据

**排查**:
1. 确认网络连接 (ping 192.168.50.1)
2. 确认 ZMQ topic 订阅地址正确
3. 确认 `topic.pyd` 和依赖 DLL 在 `utils/` 目录中
4. 确认 `os.add_dll_directory()` 正确设置

**问题**: 力传感器偏移

**解决方案**:
- V5采集: 开始采集前记下初始值作为偏置
- V7预测: 点击"调零"按钮或等待自动调零 (取最初100帧均值)

### 17.6 训练问题

**问题**: 训练 loss 不收敛

**排查**:
1. 检查数据标准化是否正确 (scaler 是否仅基于训练集)
2. 尝试调整学习率 (5e-4 → 1e-4 或 1e-3)
3. 检查是否有数据泄漏 (segment 划分是否正确)
4. 增大噪声增强 (noise_std 从 0.01 → 0.02)
5. 检查输入数据是否包含 NaN 或 Inf

**问题**: 测试集性能远差于验证集

**排查**:
1. 确认测试集文件与训练集文件完全不同
2. 检查测试集是否包含未见过的受力模式
3. 考虑增加训练数据多样性

### 17.7 HDF5 文件问题

**问题**: 文件过大

**解决方案**:
- 使用 V5_h5_edit.py 处理为 processed_*.h5 (仅保存必要数据)
- 缩短采集时间
- 压电数据可选保存 (不保存 piezo_stream)

**问题**: 处理后文件中无压电特征

**原因**: 原始 calibration_*.h5 只包含压电原始数据，需要 V5_h5_edit.py 处理后才生成 `piezo/features`。

---

## 附录 A: 命令行速查

```bash
# === V0: 数据采集准备 ===
python V0_MP4toPNG.py                           # 录制视频为PNG
python V0_PNGtoMP4.py                            # PNG转MP4
python V0_ROI.py                                 # 绘制ROI
python V0_Vision-calibration.py                  # 采集标定图像

# === V1: 参数调优 ===
python V1_Preprocess.py                          # 检测参数调整

# === V2: 匹配重建 ===
python V2_Match.py                               # 手动匹配+重建

# === V4: 实时跟踪 ===
python V4_VpTac-direction.py                     # 实时跟踪

# === V5: 多模态采集 ===
python V5_Vptac-force.py                         # 完整采集
python V5_Vptac-simple_testOK.py                 # 简化采集
python V5_h5_edit.py                             # 数据编辑

# === V6: 训练评估 ===
python V6_force_train.py --data_dir force_calibration --epochs 500
python V6_force_train.py --data_dir force_calibration --use_piezo --epochs 500
python V6_force_train_pointnet.py --data_dir force_calibration --use_piezo
python V6_force_train_lightnet.py --data_dir force_calibration --use_piezo
python V6_force_eval.py --model_dir force_calibration/model_output
python V6_force_eval_ablation.py
python V6_force_eval_cross_model.py

# === V6: 推理测试 ===
python V6_force_predict.py --model_dir force_calibration/model_output

# === V7: 实时预测 ===
python V7_Vptac-force-predict.py                 # 实时预测系统
python V7_visualize_compare.py                   # 可视化对比

# === 工具 ===
python Tactile_Finger.py                         # 压电采集GUI
python analyze_piezo_spectrum.py                 # 压电频谱分析
python compare_models.py                         # 模型对比图
```

## 附录 B: 数据目录结构示例

```
VTS4.0/
├── calibration_1mm_12X9_0512_2_Paras/    # 标定参数
│   ├── K1.txt, K2.txt, D1.txt, D2.txt
│   └── R.txt, T.txt
├── force_calibration/                     # 力标定数据
│   ├── calibration_*.h5                   # 原始采集
│   ├── ft_calibration_*.h5               # 力传感器
│   ├── processed_*.h5                     # 处理后
│   ├── model_output/                      # MLP模型
│   ├── model_output_vision_only/          # 消融: 纯视觉
│   ├── model_output_vision_piezo/         # 消融: 视觉+压电
│   ├── model_output_lightnet/             # LightNet模型
│   ├── model_output_lightnet_piezo/       # LightNet+压电
│   ├── model_output_pointnet/             # PointNet模型
│   ├── model_output_pointnet_piezo/       # PointNet+压电
│   ├── ablation_comparison/               # 消融结果
│   ├── ablation_comparison_lightnet/      # LightNet消融
│   ├── ablation_comparison_pointnet/      # PointNet消融
│   └── cross_model_comparison/            # 跨架构对比
├── recordings/                            # V7录制文件
│   └── V7_recording_*.h5
├── Vptac_shape_data_*/                    # 图像数据
│   ├── 001.png, 002.png, ...
│   └── data/
│       ├── roi_masks.npz
│       ├── matched_points.npz
│       └── marker_params.json
└── calibration_left/, calibration_right/  # 标定图像
```

---

> **文档编写**: 基于 VTS 4.0 全部源代码 | 最后更新: 2026-05-15
