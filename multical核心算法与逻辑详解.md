# multical 技术分析报告

> 基于 `/multical/multical` 源码与 `README.md` 的系统性阅读，版本 **0.4.0**（`setup.py`）。

---

## 1. 项目概述（Project Overview）

### 一句话定义

**multical 是一个多相机联合标定库与 CLI 应用，通过 Charuco / AprilGrid 标定板图像，同时估计各相机内参、相机间外参及每帧标定板位姿。**

### 展开说明

项目受 CALICO（"Calibration of Asynchronous Camera Networks"）启发，实现了一套完整的多相机标定流水线：

1. 从多相机同步拍摄的标定板图像中检测角点/标记点；
2. 单相机内参初始化（OpenCV `calibrateCameraExtended`）；
3. 基于 PnP 与图论 spanning tree 初始化多相机外参与 rig 位姿；
4. 用 **Bundle Adjustment（BA）** 非线性最小二乘优化重投影误差；
5. 导出 JSON 标定结果，并支持 Qt/PyVista 交互式可视化。

### 解决什么问题

| 问题                      | multical 的应对                                              |
| ------------------------- | ------------------------------------------------------------ |
| 多相机内参 + 外参联合标定 | 默认同一批图像同时标定内参与外参                             |
| 多标定板、多帧数据        | 支持多个 board 配置，Table 结构管理 `[camera × frame × board × point]` |
| 相机视场完全不重叠        | `is_non_overlapping` 模式：Hand-Eye 初始化（`cv2.calibrateRobotWorldHandEye`） |
| 异步/卷帘快门相机         | `rolling` 运动模型：帧内按 y 坐标插值位姿                    |
| 重复标定耗时              | 检测结果缓存为 `.detections.pkl`                             |
| 标定质量诊断              | 重投影误差报告、离群点剔除、可视化工具                       |

### 应用场景

- 多相机视觉系统（动作捕捉、立体视觉 rig、机器人感知阵列）
- 需要精确相机内外参的 3D 重建、三角测量
- 视场无重叠的多相机阵列（Hand-Eye 模式）
- 与 Kalibr 兼容的 AprilGrid 标定板

### 与传统方案相比的优势（代码体现）

1. **灵活 IO**：支持按相机分目录、自定义路径模板（`{camera}/extrinsic`）、内参/外参分离标定
2. **迭代内参 + 离群视图剔除**（`Camera.calibrate`）：自动筛选高覆盖、低误差视图，比手动选图更稳健
3. **Robust 位姿图初始化**（`align_transforms_robust` + `graph.select_pairs`）：比纯最小二乘对齐更抗 outlier（借鉴 Anipose）
4. **可配置 BA**：可选择固定/优化内参、外参、board 位姿、运动参数；支持 Huber/soft_l1 等鲁棒损失
5. **Board 非平面优化**（`adjust_board`）：允许微调 board 3D 点坐标
6. **检测缓存**：大幅加速重复实验

---

## 2. 输入与输出定义（IO Specification）

### 2.1 输入数据

#### 输入数据来源

| 来源                | 说明                                                         |
| ------------------- | ------------------------------------------------------------ |
| 图像文件            | 本地磁盘，按相机分目录存放                                   |
| 标定板配置          | YAML（`--boards` 或 `<image_path>/boards.yaml`）             |
| 已有标定（可选）    | JSON（`--calibration`），用于固定内参或初始化外参            |
| CALICO 格式（可选） | `network_specification_file.txt`（从代码推测：自动搜索上级目录） |

#### 输入数据格式

**图像目录结构（默认）：**

```
image_path/
├── cam1/
│   ├── image01.jpg
│   └── image02.jpg
├── cam2/
│   ├── image01.jpg   ← 各相机文件名需匹配（交集）
│   └── image02.jpg
└── boards.yaml       ← 可选，也可用 --boards 指定
```

**自定义路径模板：**

```bash
multical calibrate --camera_pattern '{camera}/extrinsic' --cameras cam1,cam2,cam3
```

**标定板 YAML 示例**（`example_boards/charuco_16x22.yaml`）：

```yaml
boards:
  charuco_16x22:
    _type_: charuco
    size: [16, 22]
    aruco_dict: 4X4_1000
    square_length: 0.025      # 米
    marker_length: 0.01875    # 米
    min_rows: 3
    min_points: 20
```

支持的 board 类型（`board/__init__.py`）：
- `charuco`：OpenCV ArUco CharucoBoard
- `aprilgrid`：Kalibr 风格 AprilTag 网格（依赖 `apriltags2-ethz`，Linux only）

**图像格式**：`jpg, jpeg, png, ppm, bmp`（`image/find.py`）

**是否需要预处理**

- 无需手动预处理；程序内部会：
  - 转灰度图（`cv2.IMREAD_GRAYSCALE`）
  - 检测标定板角点
  - 去畸变后 PnP 估计位姿
- 建议：图像时间同步、标定板尺寸配置与实际物理尺寸一致

#### 输入示例

README 中的典型用法：

```bash
multical calibrate --image_path ./data --boards example_boards/charuco_16x22.yaml
```

---

### 2.2 输出结果

#### 输出是什么

`multical calibrate` 默认写入 `--output_path`（未指定则用 `image_path`），文件前缀为 `--name`（默认 `calibration`）：

| 文件                         | 类型   | 含义                                               |
| ---------------------------- | ------ | -------------------------------------------------- |
| `calibration.json`           | JSON   | 最终标定结果（内参 + 外参 + 图像列表）             |
| `calibration.pkl`            | Pickle | 完整 Workspace 状态（含优化历史，供可视化/续标定） |
| `calibration.detections.pkl` | Pickle | 标定板检测缓存                                     |
| `calibration.txt`            | 文本   | 标定过程日志（从代码推测：log 文件名）             |

Hand-Eye 模式额外输出（`hand_eye/hand_eye.py`）：
- `camera_groups.pkl`
- `initial_guess.json`

#### 输出格式（JSON）

参见 `FORMAT.md` 与 `io/export_calib.py`：

```json
{
  "cameras": {
    "cam1": {
      "model": "standard",
      "image_size": [2000, 1500],
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "dist": [k1, k2, p1, p2, k3, ...]
    }
  },
  "camera_poses": {
    "cam1": { "R": [[...]], "T": [0, 0, 0] },
    "cam2_to_cam1": { "R": [[...]], "T": [...] }
  },
  "image_sets": {
    "rgb": [{ "cam1": "path/to/img.jpg", "cam2": "..." }, ...]
  }
}
```

**字段含义：**
- `K`：3×3 相机内参矩阵
- `dist`：畸变系数（长度随 model 变化：4/5/8/11/14）
- `model`：`standard | rational | thin_prism | tilted`
- `camera_poses`：相对 master 相机（默认第一个）的 4×4 位姿，分解为 R(3×3) + T(3)

#### 输出示例

见 `FORMAT.md` 第 19–62 行完整示例。

---

## 3. 整体架构设计（Architecture）

### 模块划分

```
multical/
├── app/              # CLI 入口与子命令
│   ├── multical.py   # 主入口
│   ├── calibrate.py  # 联合标定
│   ├── intrinsic.py  # 单相机内参标定
│   ├── boards.py     # 标定板生成/检测测试
│   └── vis.py        # 可视化
├── config/           # 参数定义与流水线编排
├── workspace.py      # 高层 API（核心编排类）
├── image/            # 图像加载与检测
├── board/            # 标定板定义（Charuco/AprilGrid）
├── camera.py         # 相机模型与单相机标定
├── camera_fisheye.py # 鱼眼相机模型
├── tables.py         # 多维 Table 数据结构 + 位姿初始化
├── graph.py          # 相机/板 overlap 图 + spanning tree
├── optimization/     # BA 优化核心
├── motion/           # 运动模型（静态帧 / 卷帘快门）
├── hand_eye/         # 非重叠相机 Hand-Eye 初始化
├── transform/        # 位姿/矩阵/插值工具
├── io/               # 导入导出、检测缓存、日志
└── interface/        # Qt + PyVista 可视化 GUI
```

### 各模块职责

| 模块                          | 职责                                                   |
| ----------------------------- | ------------------------------------------------------ |
| `app/`                        | CLI 解析（simple-parsing），调用 config 层编排         |
| `config/workspace.py`         | `initialise_with_images()` + `optimize()` 两阶段流水线 |
| `workspace.py`                | 状态容器：图像、检测、标定历史、导出                   |
| `image/detect.py`             | 并行加载图像、并行检测 board                           |
| `board/`                      | 物理 board 模型、2D 检测、3D 点坐标                    |
| `camera.py`                   | OpenCV 内参标定、投影、去畸变                          |
| `tables.py`                   | 点表/位姿表构建、相对位姿图估计、PnP                   |
| `optimization/calibration.py` | BA 目标函数、离群点剔除、scipy 优化                    |
| `motion/`                     | 静态/卷帘快门下的 3D→2D 投影链                         |
| `hand_eye/`                   | 无重叠视场时的相机外参初始化                           |
| `io/`                         | JSON 导入导出、pickle 缓存                             |
| `interface/`                  | 交互式 3D/2D 可视化与误差分析                          |

### 模块调用关系

```
CLI (multical calibrate)
    │
    ▼
config.workspace.initialise_with_images()
    ├── find_board_config()          → board.load_config()
    ├── find_camera_images()         → image.find.*
    ├── ws.add_camera_images()       → image.detect.load_images()
    ├── ws.detect_boards()           → image.detect.detect_images()
    ├── ws.calibrate_single()        → camera.calibrate_cameras()
    └── ws.initialise_poses()        → tables.* + hand_eye (可选)
            └── Calibration 对象创建
    │
    ▼
config.workspace.optimize()
    └── ws.calibrate()               → Calibration.adjust_outliers()
            └── bundle_adjust()      → scipy.optimize.least_squares
    │
    ▼
ws.export() + ws.dump()
```

### Pipeline 文字流程图

```
输入图像 + Board YAML
        │
        ▼
  [1] 发现相机目录 & 匹配图像文件名
        │
        ▼
  [2] 并行加载灰度图像
        │
        ▼
  [3] 并行检测标定板角点（Charuco/AprilGrid）
        │  ← 缓存到 .detections.pkl
        ▼
  [4] 构建 Point Table [cam × frame × board × point]
        │
        ▼
  [5] 单相机内参标定（迭代剔除离群视图）
        │  ← 或从已有 JSON 加载内参
        ▼
  [6] 每 (cam,frame,board) PnP 估计 board 位姿
        │
        ▼
  [7] 位姿图初始化
        │  ├─ 重叠相机：spanning tree + robust align
        │  └─ 非重叠：Hand-Eye (cv2.calibrateRobotWorldHandEye)
        ▼
  [8] 创建 Calibration 对象（内参+外参+board位姿+运动模型）
        │
        ▼
  [9] 迭代 BA（3 轮默认）：
        │  ├─ 离群点剔除（分位数阈值）
        │  └─ scipy least_squares 最小化重投影误差
        ▼
  [10] 导出 calibration.json + calibration.pkl
        │
        ▼ (可选)
  [11] multical vis → Qt/PyVista 可视化
```

---

## 4. 核心执行流程（Execution Pipeline）

### 程序入口

| 入口       | 文件                                    | 说明                              |
| ---------- | --------------------------------------- | --------------------------------- |
| CLI 主入口 | `multical/app/multical.py:cli()`        | `setup.py` 注册为 `multical` 命令 |
| 联合标定   | `multical/app/calibrate.py:calibrate()` | 子命令 `multical calibrate`       |
| 内参标定   | `multical/app/intrinsic.py`             | `multical intrinsic`              |
| 标定板工具 | `multical/app/boards.py`                | `multical boards`                 |
| 可视化     | `multical/app/vis.py`                   | `multical vis`                    |

**CLI 结构**（`multical/app/multical.py`）：

```
multical [-h] {calibrate, intrinsic, boards, show} ...
```

> 注：代码中 Vis 子命令注册为 `show` 还是从 README 的 `vis` 推断——实际 `Multical` dataclass 包含 `Vis`，simple-parsing 通常用类名小写，从代码推测子命令名为 `vis` 或通过 `--help` 确认。

### Step-by-Step 执行流程（`multical calibrate`）

#### Step 0：CLI 参数解析

- **文件**：`config/arguments.py:run_with(Calibrate)`
- **做什么**：解析 `PathOpts`、`CameraOpts`、`RuntimeOpts`、`OptimizerOpts`
- **输出**：`Calibrate` dataclass 实例

#### Step 1：创建 Workspace

- **文件**：`app/calibrate.py:calibrate()` → `workspace.Workspace(output_path, name)`
- **输入**：输出路径、标定名称
- **输出**：空 Workspace 对象

#### Step 2：配置日志

- **文件**：`io/logging.py:setup_logging()`
- **输出**：日志写入 `{output_path}/{name}.txt`

#### Step 3：加载 Board 配置

- **文件**：`config/runtime.py:find_board_config()`
- **输入**：`--boards` 或 `{image_path}/boards.yaml`
- **调用**：`board.load_config()` → OmegaConf 解析 → `CharucoBoard` / `AprilGrid`
- **输出**：`Dict[str, Board]`

#### Step 4：发现相机与图像

- **文件**：`config/runtime.py:find_camera_images()`
- **调用链**：
  - `image/find.py:find_cameras()` — 扫描目录或解析 `--cameras`
  - `image/find.py:find_images_matching()` — 取各相机文件名交集
  - 可选 `--limit_images` 随机采样
- **输出**：`struct(image_path, cameras, image_names, filenames)`

#### Step 5：初始化（`initialise_with_images`）

**5a. 加载图像**

- **文件**：`workspace.py:add_camera_images()` → `_load_images()`
- **调用**：`image/detect.py:load_images()` — 多进程并行 `cv2.imread(GRAYSCALE)`
- **输出**：`self.images[cam][frame]`，`self.image_size[cam]`

**5b. 检测标定板**

- **文件**：`workspace.py:detect_boards()`
- **调用**：`detect_boards_cached()` → `image/detect.py:detect_images()` — 多进程
  - 每帧：`board.detect(image)` → `{corners: (N,2), ids: (N,)}`
- **缓存**：`{name}.detections.pkl`（key = filenames + boards + image_sizes）
- **输出**：`point_table = tables.make_point_table()` — shape `[cam, frame, board, point]`

**5c. 单相机内参标定**

- **文件**：`workspace.py:calibrate_single()` → `camera.py:calibrate_cameras()`
- **条件**：未提供 `--calibration` 时执行
- **核心**：`Camera.calibrate()`：
  1. 收集所有 board 检测点 → `calibration_points()`
  2. 可选 `top_detection_coverage()` 限制视图数（`--limit_intrinsic`，默认 50）
  3. 循环：`cv2.calibrateCameraExtended()` → 剔除 95 分位以上误差视图
- **输出**：`List[Camera]`，每个含 K、dist、image_size

**5d. 位姿初始化**

- **文件**：`workspace.py:initialise_poses()`
- **子步骤**：
  1. `tables.make_pose_table()` — 对每个 (cam,frame,board) 调用 `cv2.solvePnPGeneric()`
  2. 若 `is_non_overlapping`：`HandEye.initialise_camera_poses()` — Hand-Eye 估计相机间变换
  3. `tables.initialise_poses()`：
     - `estimate_relative_poses(axis=0)` — 相机间相对位姿（spanning tree）
     - `estimate_relative_poses_inv(axis=2)` — board 间相对位姿
     - `relative_between_n()` — 求解 rig 时间位姿
  4. 组装 `Calibration` 对象 + `motion_model.init()`
- **输出**：`calibrations["initialisation"]`

#### Step 6：Bundle Adjustment 优化

- **文件**：`config/workspace.py:optimize()` → `workspace.py:calibrate()`
- **调用**：`Calibration.adjust_outliers(num_adjustments=3)`（默认 3 轮）
- **每轮**：
  1. `reject_outliers()` — 阈值 = `quantile(error, 0.75) × 5.0`
  2. `bundle_adjust()` — `scipy.optimize.least_squares`
     - 目标：`(reprojected - detected).ravel()` 在 inlier 上
     - 方法：`trf`，可选 loss：`linear/soft_l1/huber/arctan`
     - 稀疏 Jacobian：`sparsity_matrix`
- **可优化参数**（由 `OptimizerOpts` 控制）：
  - 内参（cameras）、外参（camera_poses）、board 位姿（board_poses）、运动（motion）、board 3D 点（boards）
- **输出**：`calibrations["calibration"]`

#### Step 7：导出

- **文件**：`workspace.py:export()` + `dump()`
- **export**：`io/export_calib.py:export_json()` → `calibration.json`
- **dump**：pickle 整个 Workspace → `calibration.pkl`

#### Step 8（可选）：可视化

- **文件**：`app/calibrate.py` 若 `--vis`；或 `multical vis --workspace_file calibration.pkl`
- **调用**：`interface/visualizer.py:visualize()` — Qt + PyVista 3D/2D 视图

---

## 5. 核心算法/逻辑说明（Core Logic）

### 5.1 重投影误差 BA（核心）

**目标函数**（`optimization/calibration.py:bundle_adjust`）：

```
minimize  Σ || π(K, dist, T_cam, T_rig, T_board, X_board) - u_detected ||²
```

其中：
- `π`：OpenCV `projectPoints`（含畸变）
- `T_cam`：相机相对 rig 的外参（`camera_poses`）
- `T_rig`：每帧 rig 位姿（`motion.frame_poses`）
- `T_board`：board 相对世界的外参（`board_poses`）
- `X_board`：board 3D 点（可优化）

**投影链**（静态模式，`motion/static_frames.py`）：

```
world_points = T_board @ X_board
local_points = T_rig @ world_points
camera_points = T_cam @ local_points
image_points = camera.project(camera_points)  # 含畸变
```

**优化器**：`scipy.optimize.least_squares(method='trf')`，带稀疏 Jacobian 加速。

### 5.2 位姿图初始化

**相机相对位姿**（`tables.py:estimate_relative_poses`）：

1. 计算 overlap 矩阵：两相机在同一 frame/board 上有效 PnP 的数量
2. `graph.select_pairs()`：贪心 spanning tree，hop_penalty=0.9 偏好短路径
3. 对每对 (parent, child)：`matrix.align_transforms_robust()` — 鲁棒 SE(3) 对齐

**Rig 位姿求解**（`tables.py:initialise_poses`）：

```
cam @ rig @ board = pose  →  rig = cam⁻¹ @ (pose @ board⁻¹)
```

### 5.3 Hand-Eye 非重叠初始化

**场景**：各相机看不到同一标定板区域。

**方法**（`hand_eye/hand_eye.py`）：
- 将 master 相机看 board M 的位姿视为 "world→gripper"
- 将 slave 相机看 board S 的位姿视为 "base→camera"
- 调用 `cv2.calibrateRobotWorldHandEye(method=CALIB_ROBOT_WORLD_HAND_EYE_SHAH)`
- 多组 board 对用 KDE 选最高密度变换（`helper.py:probabilistic_guess`）

### 5.4 迭代内参标定

**设计原因**：大数据集下手动选视图困难。

**逻辑**（`camera.py:Camera.calibrate`）：
1. 用全部有效视图标定
2. 若视图 ≥ 15：剔除 reprojection error > 95th percentile 的视图
3. 重复直到 RMS < `intrinsic_error_limit`（默认 0.5 px）

### 5.5 离群点剔除

**阈值**（`optimization/calibration.py:select_threshold`）：

```
threshold = quantile(reprojection_error, 0.75) × 5.0
```

**假设**：
- 标定板检测大部分正确
- 同步拍摄或静态场景（static 模式）
- 标定板物理尺寸配置正确

**优化点**：
- 可调整 `--iter`、`--loss`、`--outlier_threshold`
- `auto_scale` + 非线性 loss 降低大误差影响

---

## 6. 关键代码模块解析（Key Modules）

### 6.1 `multical/app/multical.py`

| 属性          | 内容                           |
| ------------- | ------------------------------ |
| 作用          | CLI 根入口，聚合 4 个子命令    |
| 关键函数      | `cli()` → `run_with(Multical)` |
| 输入/输出     | argv → 子命令 execute()        |
| Pipeline 位置 | 最顶层入口                     |

### 6.2 `multical/app/calibrate.py`

| 属性          | 内容                                            |
| ------------- | ----------------------------------------------- |
| 作用          | 联合标定子命令实现                              |
| 关键函数      | `calibrate(args)`                               |
| 流程          | Workspace 创建 → initialise → optimize → export |
| Pipeline 位置 | Step 0–7 编排                                   |

### 6.3 `multical/workspace.py`

| 属性          | 内容                                                         |
| ------------- | ------------------------------------------------------------ |
| 作用          | **高层 API 核心**，状态管理与流水线方法                      |
| 关键方法      | `add_camera_images`, `detect_boards`, `calibrate_single`, `initialise_poses`, `calibrate`, `export`, `dump`, `load` |
| 输入          | 图像路径、board 配置、优化参数                               |
| 输出          | Calibration 历史、JSON、PKL                                  |
| Pipeline 位置 | 贯穿 Step 5–7                                                |

### 6.4 `multical/config/workspace.py`

| 属性          | 内容                                     |
| ------------- | ---------------------------------------- |
| 作用          | 两阶段流水线封装                         |
| 关键函数      | `initialise_with_images()`, `optimize()` |
| Pipeline 位置 | calibrate.py 直接调用                    |

### 6.5 `multical/image/detect.py`

| 属性          | 内容                                                 |
| ------------- | ---------------------------------------------------- |
| 作用          | 图像 IO 与 board 检测                                |
| 关键函数      | `load_images()`, `detect_images()`, `detect_image()` |
| 并行          | `parmap_lists` + `multiprocessing.Pool`              |
| Pipeline 位置 | Step 5a–5b                                           |

### 6.6 `multical/board/charuco.py` & `aprilgrid.py`

| 属性          | 内容                                                         |
| ------------- | ------------------------------------------------------------ |
| 作用          | 标定板物理模型 + 2D 检测                                     |
| 关键方法      | `detect(image)`, `estimate_pose_points()`, `points`（3D 坐标） |
| Charuco       | OpenCV `detectMarkers` + `interpolateCornersCharuco`         |
| AprilGrid     | `apriltags2-ethz` + 亚像素精化                               |
| Pipeline 位置 | Step 5b                                                      |

### 6.7 `multical/camera.py`

| 属性          | 内容                                                    |
| ------------- | ------------------------------------------------------- |
| 作用          | 针孔相机模型                                            |
| 关键类/方法   | `Camera.calibrate()`, `project()`, `undistort_points()` |
| 畸变模型      | standard / rational / thin_prism / tilted               |
| Pipeline 位置 | Step 5c + BA 投影                                       |

### 6.8 `multical/tables.py`

| 属性          | 内容                                                         |
| ------------- | ------------------------------------------------------------ |
| 作用          | 多维数据结构 + 位姿初始化算法                                |
| 关键函数      | `make_point_table`, `make_pose_table`, `initialise_poses`, `estimate_relative_poses` |
| Pipeline 位置 | Step 5b–5d                                                   |

### 6.9 `multical/optimization/calibration.py`

| 属性          | 内容                                                         |
| ------------- | ------------------------------------------------------------ |
| 作用          | **BA 优化核心**                                              |
| 关键类/方法   | `Calibration.bundle_adjust()`, `adjust_outliers()`, `reprojection_error` |
| Pipeline 位置 | Step 6                                                       |

### 6.10 `multical/hand_eye/hand_eye.py`

| 属性          | 内容                                                  |
| ------------- | ----------------------------------------------------- |
| 作用          | 非重叠相机外参初始化                                  |
| 关键方法      | `initialise_camera_poses()`, `hand_eye_robot_world()` |
| Pipeline 位置 | Step 5d（条件分支）                                   |

### 6.11 `multical/motion/`

| 属性                | 内容                                         |
| ------------------- | -------------------------------------------- |
| `static_frames.py`  | 静态场景：rig 位姿不随 scanline 变化         |
| `rolling_frames.py` | 卷帘快门：帧内 y 坐标线性插值 start/end 位姿 |
| Pipeline 位置       | Step 5d 创建 + Step 6 投影                   |

### 6.12 `multical/io/export_calib.py` & `import_calib.py`

| 属性          | 内容                                  |
| ------------- | ------------------------------------- |
| 作用          | JSON 标定结果读写                     |
| 关键函数      | `export_json()`, `load_calibration()` |
| Pipeline 位置 | Step 7 + 初始化时加载已有标定         |

### 6.13 `multical/interface/visualizer.py`

| 属性          | 内容                                         |
| ------------- | -------------------------------------------- |
| 作用          | Qt GUI：3D 相机/board 可视化 + 2D 重投影叠加 |
| 依赖          | qtpy, pyvistaqt, pyvista                     |
| Pipeline 位置 | Step 8（可选）                               |

---

## 7. 依赖与外部库（Dependencies）

### 核心依赖（`setup.py:install_requires`）

| 库                                  | 作用                                                        | 为什么需要                             |
| ----------------------------------- | ----------------------------------------------------------- | -------------------------------------- |
| **numpy**                           | 数组/矩阵运算                                               | 全部几何与 Table 数据结构              |
| **opencv-contrib-python** (4.5–4.7) | 图像 IO、ArUco 检测、内参标定、PnP、投影、Hand-Eye          | 标定算法基础                           |
| **scipy**                           | `optimize.least_squares`、`spatial.transform.Rotation`、KDE | BA 优化 + 旋转平均 + Hand-Eye 密度估计 |
| **numba**                           | JIT 加速（从代码推测：transform 模块可能使用）              | 性能                                   |
| **simple-parsing**                  | CLI dataclass 参数解析                                      | 子命令与选项管理                       |
| **omegaconf**                       | YAML board 配置加载                                         | 结构化配置                             |
| **py-structs**                      | `struct`、`Table` 数据结构                                  | 多维标定数据的类型安全访问             |
| **cached-property**                 | 延迟计算 reprojection_error 等                              | 避免重复计算                           |
| **natsort**                         | 自然排序文件名                                              | 图像顺序一致性                         |
| **tqdm**                            | 进度条                                                      | 大批量图像加载反馈                     |
| **matplotlib**                      | 误差分布绘图（`plot_errors`）                               | 诊断                                   |
| **numpy-quaternion**                | 四元数运算（从代码推测：transform/interpolate）             | 位姿插值                               |
| **palettable**                      | 配色方案                                                    | 可视化                                 |
| **packaging**                       | 版本/依赖管理                                               | 工具性                                 |

### 可选依赖（`extras_require: interactive`）

| 库                          | 作用                           |
| --------------------------- | ------------------------------ |
| **qtpy**                    | Qt 抽象层，跨平台 GUI          |
| **pyvista** + **pyvistaqt** | 3D 可视化（相机、board、轨迹） |
| **qtawesome**               | GUI 图标                       |
| **colour**                  | 颜色处理                       |

### 特殊/平台依赖

| 库                  | 说明                                               |
| ------------------- | -------------------------------------------------- |
| **apriltags2-ethz** | AprilGrid 检测，**Linux only**，非 setup.py 硬依赖 |

### 按类别归纳

| 类别      | 库                                                    |
| --------- | ----------------------------------------------------- |
| 数学/优化 | numpy, scipy, numba, numpy-quaternion                 |
| 图像/检测 | opencv-contrib-python, apriltags2-ethz（可选）        |
| 深度学习  | **无**                                                |
| IO/并行   | pickle, json, multiprocessing, threading.parmap_lists |
| 可视化    | matplotlib, qtpy, pyvista, pyvistaqt                  |
| 配置/CLI  | simple-parsing, omegaconf                             |

---

## 8. 运行方式（How to Run）

### 安装

```bash
pip install multical                    # 基础功能
pip install multical[interactive]       # 含可视化
```

### 联合标定（主流程）

```bash
multical calibrate \
  --image_path ./data \
  --boards example_boards/charuco_16x22.yaml \
  --cameras cam1,cam2,cam3 \
  --output_path ./output \
  --name calibration
```

### 常用参数

| 参数                   | 默认值               | 说明             |
| ---------------------- | -------------------- | ---------------- |
| `--image_path`         | `.`                  | 图像根目录       |
| `--boards`             | 自动找 `boards.yaml` | 标定板 YAML      |
| `--cameras`            | 自动扫描子目录       | 相机名列表       |
| `--camera_pattern`     | `{camera}`           | 路径模板         |
| `--limit_images`       | 200                  | 限制帧数         |
| `--limit_intrinsic`    | 50                   | 内参标定用视图数 |
| `--distortion_model`   | standard             | 畸变模型         |
| `--motion_model`       | static               | static / rolling |
| `--is_non_overlapping` | false                | Hand-Eye 模式    |
| `--fix_intrinsic`      | false                | BA 中固定内参    |
| `--fix_camera_poses`   | false                | BA 中固定外参    |
| `--iter`               | 3                    | BA+离群剔除轮数  |
| `--loss`               | linear               | BA 损失函数      |
| `--no_cache`           | false                | 忽略检测缓存     |
| `--vis`                | false                | 标定后可视化     |

### 分离内参标定

```bash
# 1. 各相机独立内参（图像无需文件名匹配）
multical intrinsic --image_path intrinsic_images --boards boards.yaml

# 2. 固定内参做外参
multical calibrate \
  --image_path extrinsic_images \
  --calibration intrinsic_images/intrinsic.json \
  --fix_intrinsic
```

### 标定板工具

```bash
# 生成可打印 PNG
multical boards --boards example_boards/charuco_16x22.yaml \
  --paper_size A2 --pixels_mm 10 --write my_images

# 测试检测
multical boards --boards boards.yaml --detect test_image.jpg
```

### 可视化

```bash
multical vis --workspace_file calibration.pkl
```

### 常见坑

| 问题                | 原因 / 解决                                                  |
| ------------------- | ------------------------------------------------------------ |
| 检测不到 board      | 检查 YAML 尺寸（8×6 vs 6×8）、物理尺寸、用 `boards --detect` 验证 |
| 某相机无有效检测    | `check_detections` 断言失败，检查该相机图像和 board 配置     |
| 标定结果差          | 图像未同步、相机移动、视场/角度不足、畸变模型不匹配          |
| AprilGrid 不可用    | 需 Linux + `apriltags2-ethz`                                 |
| 可视化 ImportError  | `pip install multical[interactive]`                          |
| OpenCV 版本         | 锁定 4.5–4.7，更高版本可能不兼容                             |
| README 说无 fisheye | 代码已有 `camera_fisheye.py` + `--isFisheye`，但 README 可能未更新 |

---

## 9. 汇报版总结（Executive Summary）

### 一句话

**multical 是一套开源的多相机标定工具，拍一组标定板照片，自动算出所有相机的内外参数和相对位置。**

### 技术核心

1. **检测 + 单相机标定**（OpenCV）：从 Charuco/AprilGrid 图像提取角点，估计各相机焦距与畸变
2. **位姿图初始化**（图论 spanning tree + 鲁棒 SE(3) 对齐）：无需相机视场重叠也能初始化（Hand-Eye 扩展）
3. **Bundle Adjustment**（SciPy 非线性最小二乘）：联合优化所有参数，最小化重投影误差至亚像素级

### 能带来什么价值

- **降低多相机系统部署门槛**：一条命令完成内参+外参标定，输出标准 JSON
- **适配复杂场景**：多 board、非重叠视场、卷帘快门、分离内参/外参标定
- **可重复、可诊断**：检测缓存加速迭代；可视化与误差报告支持质量评估
- **工程化友好**：PyPI 安装、CLI 子命令、Python 库 API（`Workspace` 类）

### 成熟度评价

| 维度       | 评价                                                         |
| ---------- | ------------------------------------------------------------ |
| 功能完整度 | **较高** — 覆盖检测、标定、优化、导出、可视化全链路          |
| 代码质量   | **中等偏上** — 模块化清晰，依赖 `py-structs` Table 抽象，有 Hand-Eye 等高级特性 |
| 文档       | **中等** — README/FORMAT.md 有用，但部分与代码不同步（如 fisheye） |
| 生态       | **活跃维护**（v0.4.0），LGPL-3.0，有学术贡献者（CALICO/Anipose  lineage） |
| 生产就绪   | **可用于研究与原型**；生产环境建议充分验证 + 独立测试集评估重投影误差 |

---

## 附录：数据流维度说明

```
detected_points:  List[cam][frame][board] → {corners, ids}
point_table:      Table [cam, frame, board, point] → {points(2), valid}
pose_table:       Table [cam, frame, board] → {poses(4×4), valid, reprojection_error}
Calibration:
  cameras:        ParamList[Camera]           — 内参 K, dist
  camera_poses:   PoseSet [cam]               — 相机相对 rig
  board_poses:    PoseSet [board]             — board 相对世界
  motion:         MotionModel [frame]         — rig 时间位姿
```

---

*报告生成日期：2026-06-17 | 基于 multical v0.4.0 源码*
