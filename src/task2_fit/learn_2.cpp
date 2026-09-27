#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

// ================= 复用你已有的残差类 =================
struct AngleResidual {
    AngleResidual(double t, double theta) : t_(t), theta_(theta) {}

    template <typename T>
    bool operator()(const T* const p, T* residual) const {
        using std::cos;
        // θ(t) = θ0 + b·t − (A/Ω)·cos(Ωt + φ)
        T predicted = p[0] + p[1] * T(t_)
                    - (p[2] / p[3]) * cos(p[3] * T(t_) + p[4]);
        residual[0] = predicted - T(theta_);      // 预测 − 观测
        return true;
    }
    double t_, theta_;
};

// ================= 绘图参数 =================
const int W = 960, H = 720, margin = 60;

struct Range{ double xmin, xmax, ymin, ymax; };

void toPixel(double x, double y, const Range& r, int& px, int& py) {
    double nx = (x - r.xmin) / (r.xmax - r.xmin);
    double ny = (y - r.ymin) / (r.ymax - r.ymin);
    px = margin + (int)(nx * (W - 2 * margin));
    py = H - margin - (int)(ny * (H - 2 * margin));
}

int main() {
    // ---------- 观测数据 ----------
    std::vector<double> ts     = {0.0, 0.3, 0.7, 1.1, 1.6, 2.0, 2.5, 3.0, 3.4, 3.8};
    std::vector<double> thetas = {0.1, 0.5, 0.9, 1.0, 1.4, 1.5, 1.9, 2.0, 2.3, 2.4};

    // ---------- 拟合得到的最终参数 {θ0, b, A, Ω, φ} ----------
    double params[5] = {0.05, 0.55, 0.35, 2.0, -0.6};
    double* p = params;                 // p = {θ0, b, A, Ω, φ}

    // ================= ★ 方案 3 的核心：复用 AngleResidual 算残差 =================
    std::vector<double> res(ts.size());
    for (size_t i = 0; i < ts.size(); ++i) {
        AngleResidual functor(ts[i], thetas[i]);    // 用该点的 (t, θ) 构造
        functor.operator()<double>(p, &res[i]);     // 显式 <double>，残差写入 res[i]
    }
    // =============================================================================

    // ---------- 残差图的 y 范围（必须包含 0）----------
    double rmin = 1e18, rmax = -1e18;
    for (double v : res) { rmin = std::min(rmin, v); rmax = std::max(rmax, v); }
    rmin = std::min(rmin, 0.0);
    rmax = std::max(rmax, 0.0);
    double rpad = 0.15 * (rmax - rmin);
    rmin -= rpad; rmax += rpad;

    Range range_2{ts.front(), ts.back(), rmin, rmax};

    // ---------- 白底画布 ----------
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

    // ---------- 坐标轴 ----------
    cv::line(canvas, cv::Point(margin, H - margin),
             cv::Point(W - margin, H - margin), cv::Scalar(0,0,0), 2);
    cv::line(canvas, cv::Point(margin, margin),
             cv::Point(margin, H - margin), cv::Scalar(0,0,0), 2);

    // ---------- ★ 0 基准线（灰色虚线）----------
    int zx1, zy0, zx2;
    toPixel(range_2.xmin, 0.0, range_2, zx1, zy0);
    toPixel(range_2.xmax, 0.0, range_2, zx2, zy0);
    for (int x = zx1; x < zx2; x += 12) {
        cv::line(canvas, cv::Point(x, zy0),
                 cv::Point(std::min(x + 6, zx2), zy0),
                 cv::Scalar(128, 128, 128), 1);
    }

    // ---------- 画残差点 ----------
    for (size_t i = 0; i < ts.size(); ++i) {
        int px, py;
        toPixel(ts[i], res[i], range_2, px, py);
        cv::circle(canvas, cv::Point(px, py), 5, cv::Scalar(0,0,255), -1);
    }

    // ---------- 标签 ----------
    cv::putText(canvas, "residual (rad)",
                cv::Point(margin - 50, margin - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
    cv::putText(canvas, "t (s)",
                cv::Point(W - margin - 60, H - margin + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
    cv::putText(canvas, "0", cv::Point(margin - 25, zy0 + 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(128,128,128), 1);

    // ---------- 打印统计量 ----------
    double sse = 0, mean = 0;
    for (double r : res) { sse += r * r; mean += r; }
    mean /= res.size();
    double rmse = std::sqrt(sse / res.size());
    std::cout << "residual mean = " << mean << "\n";
    std::cout << "SSE = " << sse << ", RMSE = " << rmse << "\n";

    cv::imwrite("residual_plot.png", canvas);
    return 0;
}
