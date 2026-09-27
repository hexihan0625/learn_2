# vision_training —— 第二次培训三任务统一工程

> 基于 C++ / OpenCV / Eigen / Ceres 的视觉算法实践工程，包含图像处理、参数拟合两个任务，统一由根目录 CMake 构建。

## 目录

- [环境](#环境)
- [目录结构](#目录结构)
- [构建](#构建)
- [运行](#运行)
- [任务一：OpenCV 图片处理](#任务一opencv-图片处理)
- [任务二：合成旋转视频的参数拟合](#任务二合成旋转视频的参数拟合)
- [全部结果索引](#全部结果索引)
- [关键参数汇总](#关键参数汇总)
- [常见问题](#常见问题)
- [作者](#作者)
## 环境

| 项目 | 版本 / 说明 |
| ---- | ----------- |
| 操作系统 | Ubuntu 22.04 LTS |
| 编译器 | g++ 11.4.0（C++17） |
| 构建工具 | CMake 3.16+ |
| 图像处理 | OpenCV 4.x |
| 线性代数 | Eigen 3.x（仅头文件） |
| 非线性优化 | Ceres Solver 2.x |

安装依赖：

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev libeigen3-dev libceres-dev
```

验证：

```bash
pkg-config --modversion opencv4    # OpenCV 版本
ls /usr/include/eigen3/Eigen/Dense # Eigen 头文件
ls /usr/include/ceres/ceres.h      # Ceres 头文件
```

## 目录结构

```text
vision_training/
├── CMakeLists.txt                  # 根构建脚本，统一生成三个可执行程序
├── README.md                       # 本文件（总说明，含任务一分析与全部结果索引）
├── include/                        # 公共头文件
├── src/
│   ├── common/                     # 公共工具（预留）
│   ├── task1_image/
│   │   └── main.cpp                # 任务 1：OpenCV 图片处理
│   └── task2_fit/
│       └── main.cpp                # 任务 2：合成旋转视频参数拟合
├── config/                         # 配置文件（预留）
├── resources/                      # 输入素材
│   ├── test_image.jpg              # 任务 1 郁金香图片
│   └── task_2.mp4                  # 任务 2 合成视频                
└── result/                         # 全部输出结果
    ├── task1_images/               # 任务 1 结果图（17 张，含原图）
    └── task2_fit/                  # 任务 2 视频、曲线图与说明
        ├── task2_fit_result.md     # 任务 2 参数与误差说明
        ├── tracking_overlay.mp4    # 任务 2 标注视频
        ├── fit_comparison.png      # 任务 2 角度对比图
        ├── residuals.png           # 任务 2 残差图
        └── angular_velocity.png    # 任务 2 角速度图
    

> 说明：`build/` 属于构建产物，不需提交。

## 构建

```bash
cmake -S . -B build
cmake --build build -j4
```

构建后 `build/` 下生成三个可执行文件：`task1_image`、`task2_fit`、`task3_windmill`。

## 运行

```bash
# 任务 1：图片处理
./build/task1_image resources/test_image.jpg result/task1_images

# 任务 2：合成视频参数拟合（只需传入视频路径）
./build/task2_fit resources/task_2.mp4
```

> 任务 2 的输出目录固定为 `result/task2_fit/`，旋转中心固定为画面中心 `(480, 360)`，均写在源码中。
---

## 任务一：OpenCV 图片处理

**源码：** `src/task1_image/main.cpp`
**素材：** `resources/test_image.jpg`（郁金香图片）
**输出：** `result/task1_images/`（17 张结果图，含 1 张原图备份，详见本节末「输出图索引」）

### 处理流程（对应讲义五项内容）

| 步骤 | 函数 | 处理内容 | 输出文件（实际文件名） |
| ---- | ---- | -------- | -------- |
| 0 | `main` | 原图备份 | `00_original.png` |
| 1 | `step1_colorSpaces` | 读图检查、灰度图、H/S/V 单通道 | `01_gray.png`、`02_hsv_h.png`、`03_hsv_s.png`、`04_hsv_v.png` |
| 2 | `step2_filter` | 均值 / 高斯 / 中值滤波对比 | `05_mean_filter.png`、`06_gaussian_filter.png`、`07_median_filter.png` |
| 3 | `step3_extractred` | HSV 双区间红色提取 | `08_red_mask.png` |
| 4 | `step4_shape` | 形态学四操作 + 轮廓筛选 + 外接矩形 | `09_erode.png`、`10_dilate.png`、`11_open.png`、`12_close.png`、`13_contours.png` |
| 5 | `step5_draw_transform` | 绘制 + 旋转 35° + 裁剪左上 1/4 | `14_draw.png`、`15_rotate35.png`、`16_crop_topleft.png` |

### 1. 读图与颜色转换

```cpp
cv::Mat gray;
cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);   // 灰度图：单通道亮度

cv::Mat hsv;
cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);     // 转 HSV

cv::Mat hsvChannels[3];
cv::split(hsv, hsvChannels);                   // 拆 H / S / V
```

**分析：**

- OpenCV 读入的彩色图默认是 **BGR** 顺序。转灰度按人眼敏感度加权：`Gray = 0.114·B + 0.587·G + 0.299·R`，绿色权重最高、蓝色最低，更符合人眼观感；灰度图为单通道，数据量降为 1/3。
- HSV 把颜色拆成三个独立维度：**H 色相**（0~179，OpenCV 为 360° 的一半）、**S 饱和度**（0~255，颜色鲜艳程度）、**V 明度**（0~255，亮度）。
- **为什么提色要用 HSV**：BGR 空间下光照变化会同时改变三通道值，颜色难以用固定范围描述；HSV 把"颜色（H）"与"亮度（V）"分离，只锁定 H 区间即可稳定提取某种颜色，这是第 3 步红色提取的基础。

**观察要点：** `gray.png` 仅保留明暗；`hsv_h.png` 反映颜色分布；`hsv_s.png` 越白表示越鲜艳；`hsv_v.png` 反映亮度分布。

### 2. 滤波对比

```cpp
cv::blur(img, blurImg, cv::Size(CORE, CORE));               // 均值滤波，核 15x15
cv::GaussianBlur(img, gaussianImg, cv::Size(CORE, CORE), 0);// 高斯滤波，sigmaX 自动
cv::medianBlur(img, medianImg, CORE);                       // 中值滤波，核 15
```

| 滤波 | 原理 | 特点 |
| ---- | ---- | ---- |
| 均值 `blur` | 邻域平均值 | 最快，边缘最模糊 |
| 高斯 `GaussianBlur` | 高斯加权平均 | 过渡自然，最常用 |
| 中值 `medianBlur` | 邻域中位数 | 去椒盐噪声强，保边较好 |

**参数记录：** 核尺寸均为 `CORE = 15`；高斯 `sigmaX = 0` 表示由核尺寸自动推导。**核越大平滑越强，也越容易丢失花瓣窄边缘等细节。**

### 3. 红色提取

```cpp
cv::inRange(hsv, cv::Scalar(0,   100, 50), cv::Scalar(10,  255, 255), mask1); // 区间 1
cv::inRange(hsv, cv::Scalar(170, 100, 50), cv::Scalar(179, 255, 255), mask2); // 区间 2
cv::add(mask1, mask2, redMask);                                              // 合并
```

**阈值记录：** `H = 0–10 或 170–179`，`S >= 100`，`V >= 50`。

**分析：**

1. **红色为什么要分两段**：HSV 色相是环形的，红色跨在 0°/360° 交界处（OpenCV 中为 0 与 179 附近），只取一段会漏掉一半红色，故用两段再合并。
2. **S、V 下限的作用**：`S >= 100` 过滤灰白低饱和区域，`V >= 50` 过滤过暗区域，避免误判。
3. `inRange` 输出**单通道二值掩膜**（255 = 命中，0 = 未命中）。
4. **红色、黄色边缘与阴影的处理效果说明**：
   - **红色区域**：被较完整地提取为白色掩膜，是后续轮廓提取的主要目标；
   - **黄色边缘**：黄色的 H 约为 20~35，**不在** 0–10 / 170–179 区间内，因此黄色花瓣边缘基本不被提取，只可能因饱和度、亮度变化在交界处产生零星像素；
   - **阴影区域**：阴影使 V（明度）降低，`V >= 50` 可过滤掉大部分暗部；但红色物体上的阴影若 V 仍高于 50 且 S 足够，会被一并提取，表现为红色区域边缘的缺口或碎片。
   - 结论：该双区间阈值对**纯红**响应良好，对黄色与深阴影有天然过滤，但仍需依靠第 4 步形态学与面积筛选进一步清理碎片。
5. 程序用 `countNonZero` 统计掩膜白色像素占比，输出"红色区域占比"，用于量化说明红色在画面中的大小。

### 4. 形态学与轮廓

```cpp
const int K = 7;
cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(K, K));

cv::erode(mask,  eroded,  kernel);                         // 腐蚀：前景缩小
cv::dilate(mask, dilated, kernel);                         // 膨胀：前景扩大
cv::morphologyEx(mask, opened, cv::MORPH_OPEN,  kernel);   // 开：去小白噪点
cv::morphologyEx(mask, closed, cv::MORPH_CLOSE, kernel);   // 闭：填补小黑洞、连接断裂
```

- 结构元素：`MORPH_ELLIPSE` 椭圆 7×7；以**白色像素为前景**。
- 选择**闭运算结果**提取外轮廓（断裂更少、轮廓更完整）。

```cpp
cv::findContours(closed.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
// RETR_EXTERNAL 只取最外层轮廓；CHAIN_APPROX_SIMPLE 压缩直线段冗余点

for (auto& c : contours) {
    double area = cv::contourArea(c);
    if (area < 500.0) continue;                    // 面积筛选，丢弃噪声
    cv::drawContours(canvas, contours, i, cv::Scalar(0, 255, 0), 2);   // 绿色轮廓
    cv::Rect box = cv::boundingRect(c);                                // 轴对齐外接矩形
    cv::rectangle(canvas, box, cv::Scalar(0, 0, 255), 2);              // 红色矩形
    // 标注序号与筛选后的轮廓面积，如 "#0 A=1234"
}
```

**筛选参数：** 面积阈值 `AREA_THRESHOLD = 500`（像素面积），小于该值的轮廓视为噪声丢弃。

> **框选对象说明**：按讲义要求，框选对象是**红色连通区域**，不要求一个框恰好对应一朵完整的花。

### 5. 绘制与变换

```cpp
cv::circle(canvas, cv::Point(200, 200), 80, cv::Scalar(0, 0, 255), -1);         // 实心红圆
cv::rectangle(canvas, cv::Rect(500, 150, 300, 200), cv::Scalar(0, 255, 0), 3);  // 绿色空心矩形
cv::putText(canvas, "RoboMaster", cv::Point(60, 500),
            cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 255, 0), 3);         // 黄色文字

cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
cv::Mat rotMat = cv::getRotationMatrix2D(center, 35.0, 1.0);   // 绕中心 35°，缩放 1.0
cv::warpAffine(img, rotated, rotMat, img.size());              // 输出仍为原图尺寸

cv::Rect roi(0, 0, img.cols / 2, img.rows / 2);                // 左上角 1/4
cv::Mat cropped = img(roi);
```

| 项目 | 参数 | 说明 |
| ---- | ---- | ---- |
| 圆 | 圆心 (200,200)、半径 80、BGR(0,0,255)、线宽 -1 | -1 表示填充 |
| 矩形 | `Rect(500,150,300,200)`、BGR(0,255,0)、线宽 3 | 参数是 (x, y, 宽, 高) |
| 文字 | (60,500)、`FONT_HERSHEY_SIMPLEX`、缩放 2.0、线宽 3 | 不支持中文 |
| 旋转 | 中心 = 图像中心、35°、缩放 1.0 | 正角度为逆时针 |
| 裁剪 | `Rect(0,0, cols/2, rows/2)` | 左上角四分之一 |

**要点：**

- `Scalar` 是 **BGR** 顺序，红色写 `(0,0,255)`。
- 线宽 **-1 表示填充**，正数表示线宽。
- `putText` **不支持中文**，只能用英文/数字。
- 旋转必须经 `getRotationMatrix2D` 指定中心，否则会绕左上角原点旋转；输出尺寸仍用 `img.size()`，故**四角被裁掉**（本任务未要求保留全图）。
- `img(roi)` 是**视图**操作，不复制数据。

### 输出图索引（`result/task1_images/`，共 17 张）

> 文件名均带两位序号前缀，方便按处理顺序浏览；`main` 最后额外导出原图便于对照。

| 序号 | 文件名 | 来源 | 内容说明 | 观察要点 |
| ---- | ------ | ---- | -------- | -------- |
| 00 | `00_original.png` | 原图备份 | 未经处理的原始输入图 | 作为后续所有结果的对照基准 |
| 01 | `01_gray.png` | 1 颜色空间 | BGR → 灰度图（单通道） | 只剩明暗层次，无颜色信息 |
| 02 | `02_hsv_h.png` | 1 颜色空间 | HSV 的 H（色相）通道 | 不同颜色呈不同灰度，反映颜色分布 |
| 03 | `03_hsv_s.png` | 1 颜色空间 | HSV 的 S（饱和度）通道 | 越白 = 颜色越鲜艳，灰色低饱和区域偏暗 |
| 04 | `04_hsv_v.png` | 1 颜色空间 | HSV 的 V（明度）通道 | 反映亮度分布，与光照强弱一致 |
| 05 | `05_mean_filter.png` | 2 滤波 | 均值滤波（核 15×15） | 整体变模糊，边缘过渡最生硬 |
| 06 | `06_gaussian_filter.png` | 2 滤波 | 高斯滤波（核 15×15） | 模糊程度接近均值但过渡更自然 |
| 07 | `07_median_filter.png` | 2 滤波 | 中值滤波（核 15） | 边缘保留最好，噪点被去除 |
| 08 | `08_red_mask.png` | 3 红色提取 | 双区间红色二值掩膜 | 白色 = 命中的红色区域，黑色 = 非红 |
| 09 | `09_erode.png` | 4 形态学 | 腐蚀（椭圆核 7×7） | 白色前景整体缩小、细碎处消失 |
| 10 | `10_dilate.png` | 4 形态学 | 膨胀（椭圆核 7×7） | 白色前景扩大、断裂处被连接 |
| 11 | `11_open.png` | 4 形态学 | 开运算 = 先腐蚀后膨胀 | 去掉小白噪点，主体大小基本不变 |
| 12 | `12_close.png` | 4 形态学 | 闭运算 = 先膨胀后腐蚀 | 填补内部小黑洞，用于取轮廓 |
| 13 | `13_contours.png` | 4 轮廓 | 闭运算结果上提取的外轮廓 | 红色轮廓线 + 绿色外接矩形 + `#序号 A=面积` 标注 |
| 14 | `14_draw.png` | 5 绘制 | 圆 / 矩形 / 文字 | 实心红圆、绿色空心矩形、黄色 `RoboMaster` 文字 |
| 15 | `15_rotate35.png` | 5 变换 | 绕图像中心逆时针旋转 35° | 输出仍为原尺寸，故四角被裁掉 |
| 16 | `16_crop_topleft.png` | 5 变换 | 裁剪原图左上 1/4 | 尺寸为原图的一半（480×360） |

**快速定位：**

- 想看**颜色空间**拆解 → `01`~`04`
- 想看**滤波差异** → `05`~`07` 三张互相对比着看
- 想看**红色提取** → `08`
- 想看**形态学效果** → `09`~`12`，其中 `11`（开）与 `12`（闭）最常对比
- 想看**轮廓筛选** → `13`
- 想看**绘制与几何变换** → `14`~`16`

---

## 任务二：合成旋转视频的参数拟合

**源码：** `src/task2_fit/main.cpp`
**素材：** `resources/task_2.mp4`（960×720，60 FPS，24 秒，1440 帧）
**输出：** `result/task2_fit/`
**参数说明：** `result/task2_fit_result.md`

### 模型与待估参数

讲义给出的角速度模型：

```text
ω(t) = b + A·sin(Ωt + φ)
```

积分得到角度模型（本任务拟合采用的形式）：

```text
θ(t) = θ0 + b·t − (A/Ω)·cos(Ωt + φ)
```

| 参数 | 含义 | 单位 | 约束 |
| ---- | ---- | ---- | ---- |
| `θ0` | 初始角度（第 0 帧） | rad | — |
| `b` | 平均角速度 | rad/s | — |
| `A` | 角速度振幅 | rad/s | `A > 0` |
| `Ω` | 速度变化的频率参数 | rad/s | `Ω > 0` |
| `φ` | 相位 | rad | 报告时归一化到 `[−π, π)` |

**角度定义：** 图像 x 向右、y 向下；从右方起算、逆时针为正：

```text
θ_wrapped,i = atan2(c_y − y_i, x_i − c_x)
```

**已知条件：** 旋转中心 (480, 360)、半径 220 像素（可直接使用，也可通过图像检测中心）。

### 方法

1. **目标提取**：BGR → HSV，`inRange` 提取青色目标（H 约 80~105），形态学开运算去噪，取最大连通域质心作为目标中心。
2. **角度计算与展开**：按上式用 `atan2` 求每帧折返角度，再逐帧叠加"最短角差"展开为连续序列（避免圈数丢失）。
3. **初值估计**（关键）：角度模型中 `b` 与 `(A, Ω)` 强耦合，Ω 初值偏离会导致求解器把 Ω 推向高频、A 压到 0 的病态解。因此先对 `θ ≈ b·t + c` 做线性拟合得到 `b` 与截距，对去趋势残差做**频率扫描（DFT）**找到主频估计 `Ω`，再由残差振幅 `(A/Ω)` 反推 `A`。
4. **Ceres 拟合**：以连续角度 `θ(t)` 为观测，用 `AutoDiffCostFunction` 对 5 个参数自动求导，`DENSE_QR` 线性求解器；约束 `A > 0`、`Ω ∈ [π/span, π/(2·dt)]`（上下界防止高频病态解）。

> **为什么拟合角度而不是角速度**：角度可由每帧中心直接观测，拟合角度能同时约束 `θ0` 与相位，使曲线与观测对齐；若只拟合角速度则不需估计 `θ0`。

### 实测结果（task_2.mp4）

检测成功率 **1440/1440 = 100%**，Ceres 5 次迭代收敛（最终代价 0.00399）：

| 参数 | 估计值 | 单位 |
| ---- | ------ | ---- |
| `theta0` | 0.6046 | rad |
| `b` | **1.3500** | rad/s |
| `A` | **0.5499** | rad/s |
| `Omega` | **1.6499** | rad/s |
| `phi` | 0.7029 | rad |
| 速度变化周期 `T = 2π/Ω` | 3.808 | s |

**误差指标：** 角度 RMSE = **0.002353 rad = 0.1348°**；有效样本 1440 帧，帧范围 [0, 23.98] s。

**验证：** 角速度范围 [0.800, 1.900] rad/s，符合 `b ± A`（1.3500 ± 0.5499）；拟合曲线与观测点几乎完全重合。

> **初值为何重要（对比）**：若 Ω 初值取 1.0 且不设上界，求解器会收敛到 Ω=1024.7、A≈0 的病态解，角度 RMSE 高达 13.31°；采用 DFT 主频估计初值后，RMSE 降至 0.135°，降幅约 99 倍。

### 输出文件

| 文件 | 说明 |
| ---- | ---- |
| `tracking_overlay.mp4` | 合成视频上标记目标中心、中心连线与角度 |
| `fit_comparison.png` | 角度观测点（红）与拟合曲线（蓝）对比 |
| `angular_velocity.png` | 估计的角速度曲线 `ω(t)`，并标出平均角速度 `b` |
| `residuals.png` | 角度残差曲线，并画出零基准线 |
| `task2_fit_result.md` | 模型、参数、方法、误差指标 |

---

## 全部结果索引

工程运行后，`result/` 下共产生 **21 个交付文件**，按任务归类如下。

### 任务一：`result/task1_images/` —— 17 张 PNG

| 序号 | 文件名 | 对应步骤 |
| ---- | ------ | -------- |
| 00 | `00_original.png` | 原图备份 |
| 01 | `01_gray.png` | 1 颜色空间 |
| 02 | `02_hsv_h.png` | 1 颜色空间 |
| 03 | `03_hsv_s.png` | 1 颜色空间 |
| 04 | `04_hsv_v.png` | 1 颜色空间 |
| 05 | `05_mean_filter.png` | 2 滤波 |
| 06 | `06_gaussian_filter.png` | 2 滤波 |
| 07 | `07_median_filter.png` | 2 滤波 |
| 08 | `08_red_mask.png` | 3 红色提取 |
| 09 | `09_erode.png` | 4 形态学 |
| 10 | `10_dilate.png` | 4 形态学 |
| 11 | `11_open.png` | 4 形态学 |
| 12 | `12_close.png` | 4 形态学 |
| 13 | `13_contours.png` | 4 轮廓 |
| 14 | `14_draw.png` | 5 绘制 |
| 15 | `15_rotate35.png` | 5 变换 |
| 16 | `16_crop_topleft.png` | 5 变换 |

### 任务二：`result/task2_fit/` —— 4 个结果文件

| 文件名 | 类型 | 说明 |
| ------ | ---- | ---- |
| `task2_fit_result.md` | 文档 | 模型、参数、方法、误差指标 |
| `tracking_overlay.mp4` | 视频 | 标注了目标中心、中心连线与角度的合成视频 |
| `fit_comparison.png` | 图片 | 角度观测点与拟合曲线对比 |
| `residuals.png` | 图片 | 逐帧角度残差 |
| `angular_velocity.png` | 图片 | 角速度曲线 `ω(t)` |

> 结果文件均由程序自动写入，重复运行会覆盖同名文件；`result/` 目录须预先存在。

---

## 关键参数汇总

### 任务一（`src/task1_image/main.cpp`）

| 参数 | 取值 | 位置 |
| ---- | ---- | ---- |
| 滤波器核尺寸 `CORE` | 15 | 全局常量 |
| 红色提取色相区间 | `H ∈ [0,10] ∪ [170,179]`，`S ≥ 100`，`V ≥ 50` | `step3_extractred` |
| 形态学核 | `MORPH_ELLIPSE` 7×7 | `step4_shape` |
| 轮廓面积阈值 | 500 像素 | `step4_shape` |
| 旋转角度 | 35°（绕图像中心，逆时针） | `step5_draw_transform` |
| 裁剪区域 | `Rect(0, 0, cols/2, rows/2)` | `step5_draw_transform` |

### 任务二（`src/task2_fit/main.cpp`）

| 参数 | 取值 | 说明 |
| ---- | ---- | ---- |
| 旋转中心 | `(480, 360)` | 画面中心 |
| 参考圆半径 | 220 像素 | 叠加标注用 |
| 青色掩膜阈值 | `H ∈ [80,105]`，`S,V ≥ 80` | `detectCyanCenter` |
| 频率扫描点数 | 4000（对数均匀） | 初值估计 |
| 扫描区间 | `[π/span, π/(2·dt)]` | 由录像时长与采样间隔决定 |
| Ceres 线性求解器 | `DENSE_QR` | 5 维稠密小问题 |
| 最大迭代次数 | 200 | 安全上限，实际 5 次收敛 |
| 参数约束 | `A > 0`，`Ω ∈ [π/span, π/(2·dt)]` | 防止病态解 |
| 输出视频编码 | `mp4v` | 兼容性见「常见问题」 |

---

## 常见问题

| 问题 | 原因 | 解决 |
| ---- | ---- | ---- |
| 找不到 `opencv2/opencv.hpp` | 未装 OpenCV | `sudo apt install libopencv-dev` |
| 找不到 `<ceres/ceres.h>` | 未装 Ceres | `sudo apt install libceres-dev` |
| 找不到 `<Eigen/Dense>` | 未装 Eigen | `sudo apt install libeigen3-dev` |
| CMake 报找不到 Eigen3 | 需指定 `NO_MODULE` | 使用 `find_package(Eigen3 REQUIRED NO_MODULE)` |
| 图片读入失败 | 路径错误 | 确认 `resources/test_image.jpg` 存在 |
| 输出视频打不开/为空 | 输出目录不存在或编码不支持 | 先建目录，编码用 `mp4v` |
| 中文显示成方块 | `putText` 不支持中文 | 画面文字仅用英文/数字 |
| 颜色画反（红变蓝） | `Scalar` 是 BGR 顺序 | 红色写 `(0,0,255)` |
| 任务 2 拟合结果偏离较大 | 初值/阈值不合适或观测含异常 | 检查青色阈值与检测成功率，必要时改用鲁棒损失 |
| 任务 2 生成视频无法预览 | 浏览器只认 H.264 编码，`mp4v` 不兼容 | 用 ffmpeg 转码为 H.264（见下方命令） |

```bash
# 安装 ffmpeg
sudo apt install ffmpeg

# 验证安装
ffmpeg -version

# 把 mp4v 编码的输出转为浏览器兼容的 H.264
ffmpeg -i result/task2_fit/tracking_overlay.mp4 \
       -c:v libx264 -pix_fmt yuv420p \
       result/task2_fit/tracking_overlay_h264.mp4
```

## 作者

| 项目 | 信息 |
| ---- | ---- |
| 姓名 | 何熙涵 |
| 学号 | 2010718320 |
| 日期 | 2026.09.25 |
