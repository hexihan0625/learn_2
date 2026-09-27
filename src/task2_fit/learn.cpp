cpp

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

// ============ 绘图参数 ============
const int    W = 960;          // 宽
const int    H = 720;          // 高
const int    margin = 60;      // 四周边距

// ============ 映射函数（复用你之前写的 toPixel）============
struct Range { double xmin, xmax, ymin, ymax; };

void toPixel(double x, double y, const Range& r, int& px, int& py) {
    double nx = (x - r.xmin) / (r.xmax - r.xmin);
    double ny = (y - r.ymin) / (r.ymax - r.ymin);
    px = margin + (int)(nx * (W - 2 * margin));
    py = H - margin - (int)(ny * (H - 2 * margin));   // y 轴翻转
}

int main() {
    // ---------- 观测数据（示例：你自己从拟合结果/文件读进来）----------
    std::vector<double> ts    = {0.0, 0.3, 0.7, 1.1, 1.6, 2.0, 2.5, 3.0, 3.4, 3.8};
    std::vector<double> thetas= {0.1, 0.5, 0.9, 1.0, 1.4, 1.5, 1.9, 2.0, 2.3, 2.4};

    // ---------- 拟合出来的五个参数（示例值，换成你的真实结果）----------
    double theta0 = 0.05, b = 0.55, A = 0.35, Omega = 2.0, phi = -0.6;

    // ========== 步骤 1：建画布（白底）==========
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

    // ========== 步骤 2：确定数据范围 ==========
    double xmin = ts.front();
    double xmax = ts.back();

    double ymin = 1e18, ymax = -1e18;
    // 观测值的范围
    for (double v : thetas) { ymin = std::min(ymin, v); ymax = std::max(ymax, v); }
    // 拟合值的范围（在时间轴上均匀采样，算一遍模型值）
    for (int j = 0; j <= 1000; ++j) {
        double t = xmin + (xmax - xmin) * j / 1000.0;
        double y = theta0 + b * t - (A / Omega) * std::cos(Omega * t + phi);
        ymin = std::min(ymin, y);
        ymax = std::max(ymax, y);
    }
    // 留 5% 边距，防止点/线贴边被裁掉
    double pad = 0.05 * (ymax - ymin);
    ymin -= pad;
    ymax += pad;

    Range range{xmin, xmax, ymin, ymax};

    // ========== 步骤 3：画观测点（实心小圆）==========
    for (size_t i = 0; i < ts.size(); ++i) {
        int px, py;
        toPixel(ts[i], thetas[i], range, px, py);
        cv::circle(canvas, cv::Point(px, py), 4,
                   cv::Scalar(0, 0, 255), -1);   // 红色实心（BGR）
    }

    // ========== 步骤 4：画拟合曲线（均匀采样 + 连线）==========
    int prev_px = 0, prev_py = 0;
    for (int j = 0; j <= 800; ++j) {
        double t = xmin + (xmax - xmin) * j / 800.0;         // ★ 均匀取 801 个点
        double y = theta0 + b * t - (A / Omega) * std::cos(Omega * t + phi);  // 复用模型
        int px, py;
        toPixel(t, y, range, px, py);
        if (j > 0)
            cv::line(canvas, cv::Point(prev_px, prev_py), cv::Point(px, py),
                     cv::Scalar(255, 0, 0), 2);              // 蓝色曲线
        prev_px = px; prev_py = py;                          // 首点不连线，之后逐段连
    }

    // ========== 步骤 5：坐标轴 + 标签 ==========
    cv::line(canvas, cv::Point(margin, H - margin), cv::Point(W - margin, H - margin),
             cv::Scalar(0, 0, 0), 1);                        // x 轴
    cv::line(canvas, cv::Point(margin, margin), cv::Point(margin, H - margin),
             cv::Scalar(0, 0, 0), 1);                        // y 轴

    cv::putText(canvas, "t (s)", cv::Point(W - margin - 60, H - margin + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(canvas, "theta (rad)", cv::Point(margin - 50, margin - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // 图例
    cv::circle(canvas, cv::Point(W - 200, 40), 5, cv::Scalar(0, 0, 255), -1);
    cv::putText(canvas, "observed", cv::Point(W - 185, 46),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    cv::line(canvas, cv::Point(W - 200, 70), cv::Point(W - 170, 70),
             cv::Scalar(255, 0, 0), 2);
    cv::putText(canvas, "fitted", cv::Point(W - 160, 76),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    // ========== 保存 ==========
    cv::imwrite("fit_comparison.png", canvas);
    return 0;}