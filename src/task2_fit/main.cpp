#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include <string>

//声明pi
const double PI = M_PI;
//定义残差
struct AngleResidual {
    AngleResidual(double t, double theta):t_(t), theta_(theta){}
    template < typename T >
    bool operator()(const T*const p,T*residual)const{
        using std::cos;
        // θ(t) = θ0 + b·t − (A/Ω)·cos(Ωt + φ)
        // p = { θ0, b, A, Ω, φ }
        T predicted = p[0] + p[1] * T(t_)
                    -(p[2]/p[3])*cos(p[3]*T(t_) + p[4]);
        residual[0] = predicted - T(theta_);                // 预测 − 观测
        return true; 
    }
    double t_, theta_;
};
//角度展开
static double unwrapAngle(double prev, double curr)
{
    double diff = curr - prev;
    while (diff >=  M_PI) diff -= 2.0 * M_PI;  // 规范到 [-π, π)
    while (diff <  -M_PI) diff += 2.0 * M_PI;
    return prev + diff;                         // 累加最短角差
}
//模板4
static bool detectCyanCenter(const cv::Mat& frame, cv::Point2f& centerOut,
                             cv::Mat* maskOut = nullptr)
{
    //提取青色
    cv::Mat hsv, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(80, 80, 80),
                     cv::Scalar(105, 255, 255),
                     mask);
    //去噪
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    /*if (maskOut) *maskOut = mask.clone();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return false;
    */

    //筛选轮廓
    std::vector<std::vector<cv::Point>>contours;
    cv::findContours(mask, contours, 
                    cv::RETR_EXTERNAL,        
                    cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()){return false;}  
    //面积筛选
    size_t best = 0;
    double bestArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i){
        double a = cv::contourArea(contours[i]);
        if(a > bestArea){ bestArea = a; best = i; }       
    }
    if (bestArea < 10.0){return false; }
    //质心
    cv::Moments m = cv::moments(contours[best]);
    if (m.m00 == 0.0){return false;}
    centerOut = cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
    return true;
}
//画面数据
const int    W = 960, H = 720;
const int    margin = 60;

struct Range { double xmin, xmax, ymin, ymax; };
//映射函数
void toPixel(double x, double y, const Range& r, int& px, int& py) {
    px = margin + (int)((x - r.xmin) / (r.xmax - r.xmin) * (W - 2 * margin));
    py = H - margin - (int)((y - r.ymin) / (r.ymax - r.ymin) * (H - 2 * margin));
}

//坐标轴
static void drawxy(cv::Mat& img){
    cv::line(img, cv::Point(margin, H - margin),
             cv::Point(W - margin, H - margin), 
             cv::Scalar(0,0,0), 2);
    cv::line(img, cv::Point(margin, margin),
             cv::Point(margin, H - margin), 
             cv::Scalar(0,0,0), 2);
}

int main(int argc, char** argv)
{
    /*// 默认视频路径：优先用编译时注入的源码目录，保证在 build/ 里运行也能找到
    std::string defaultPath = "resources/task_2.mp4";
    defaultPath = std::string(TASK2_SOURCE_DIR) + "/resources/task_2.mp4";
    const std::string inputPath = (argc >= 2) ? argv[1] : defaultPath;*/
    //打开视频
    const std::string inputPath = argv[1];
    cv::VideoCapture cap("resources/task_2.mp4");  

    if (!cap.isOpened()){
        std::cout<<"无法打开视频！路径："<<inputPath<<std::endl;//相对路径
        return -1;
    }
    //视频参数
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 1e-6) {
        fps = 60.0;   // 读不到就假定 60，避免除零
    }
    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);//1440
    int width       = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height      = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    //声明变量
    const double cx = 480.0,cy = 360.0;
    std::vector<double> ts, thetas;
    //std::vector<Observation> obs;

    int frameIdx = 0;
    double prevRawTheta = 0.0, /*prevUnwrapped = 0.0,*/ unwrappedTheta = 0.0;
    bool firstValid = true;

    cv::Mat frame;
    //循环得到视频每一帧
    while (cap.read(frame)){
        double t = frameIdx/fps;
        
        cv::Point2f centerPt;
        if (detectCyanCenter(frame, centerPt)) {   // ← 板块4的调用 
            // 循环内：
            //  这里放板块3的展开逻辑（if firstValid / else）
            const double rawTheta = std::atan2(cy - centerPt.y, centerPt.x - cx);
            if (firstValid) {
                //prevUnwrapped  = rawTheta; 
                unwrappedTheta = rawTheta; 
                prevRawTheta = rawTheta; 
                firstValid = false; 
            }else{
                double d = rawTheta - prevRawTheta;
                while (d >= M_PI)  d -= 2.0 * M_PI;
                while (d < -M_PI)  d += 2.0 * M_PI;
                unwrappedTheta += d;
                prevRawTheta  = rawTheta;
                //prevUnwrapped = unwrappedTheta;
            }  
            ts.push_back(t);
            thetas.push_back(unwrappedTheta); 
            //  展开成功后：ts.push_back(t); thetas.push_back(unwrappedTheta);
        }    
        ++frameIdx;
    }
    cap.release();
    //判断帧数
    size_t N = ts.size();
    if (N < 10){
        return false;
    }
    //线性最小二乘估计
    Eigen::MatrixXd design(N, 2);
    Eigen::VectorXd y(N);
    for (size_t i = 0; i < N; ++i){
        design(i, 0) = ts[i];
        design(i, 1) = 1.0;
        y(i) = thetas[i];
    }
    //OR求解最小二乘
    Eigen::Vector2d bc = design.colPivHouseholderQr().solve(y);
    double b_init = bc[0];
    double c_init = bc[1];

    //估计欧
    std::vector<double>detr(N);
    for (size_t i = 0; i < N; ++i){
        detr[i] = thetas[i] - (b_init * ts[i] + c_init);       
    }
    // 频率扫描范围
    const double t0 = ts.front();
    const double tN = ts.back();

    double span     = std::max(1e-6, tN - t0);
    double dtAvg    = span / std::max<size_t>(1, N - 1);
    double omegaLo  = PI / span;             // 最慢
    double omegaHi  = PI / (2.0 * dtAvg);    // 最快（接近采样极限）
    //频率扫描
    const int N_SCAN = 4000;
    double bestOmega = 1.0, bestMag = -1.0;

    for (int k = 0; k < N_SCAN; ++k){
        double r =static_cast<double>(k) / (N_SCAN - 1);
        //对数均匀
        double omega = omegaLo * std::pow(omegaHi / omegaLo, r);
        double re = 0.0, im = 0.0;
        for (size_t i = 0; i < N; ++i){
            double ph = omega * ts[i];
            re += detr[i] * std::cos(ph);
            im -= detr[i] * std::sin(ph);
        }
        double mag = re*re + im*im;
        if (mag > bestMag){
            bestMag = mag; 
            bestOmega = omega;
        }
    }
    // 残差振幅 ≈ A/Ω，反推 A
    double detMin      = *std::min_element(detr.begin(), detr.end());
    double detMax      = *std::max_element(detr.begin(), detr.end());
    double ampResidual = 0.5 * (detMax - detMin);
    double A_init      = std::max(1e-3, ampResidual * bestOmega);

    //建立问题与约束
    double params[5] = {c_init, b_init, A_init, bestOmega, 0.0};
    ceres::Problem problem;
    for (size_t i = 0; i < N; ++i){
         ceres::CostFunction*cost=
         new ceres::AutoDiffCostFunction<AngleResidual, 1, 5>(
          new AngleResidual(ts[i], thetas[i]));
        problem.AddResidualBlock(cost, /*new ceres::HuberLoss(1.0)*/ nullptr, params);
    }
    //按参数块内分量索引设置上下界
    problem.SetParameterLowerBound(params, 2, 1e-6);    // A > 0
    problem.SetParameterLowerBound(params, 3, omegaLo); // Ω 下界
    problem.SetParameterUpperBound(params, 3, omegaHi); // Ω 上界（防抽象解）

    //优化
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR; //3.12.小规模稠密问题常用起点
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;
    //计算
    ceres::Solve(options, &problem, &summary);//problem
    //打印摘要报告
    std::cout << summary.BriefReport() << "\n";   // 讲义 3.5

    //角度模型
    const double theta0 = params[0];
    const double b      = params[1];
    const double A      = params[2];
    const double Omega  = params[3];
          double phi    = params[4];
    //归一化
    while ( phi > PI ){phi -= 2.0*PI;}
    while ( phi < -PI ){phi += 2.0*PI;}

    auto angleModel = [&](double t){
        return theta0+b*t-(A/Omega)*std::cos(Omega*t + phi);
    };
    double sse = 0.0;
    std::vector<double>residuals;
    for (size_t i = 0; i < N; ++i) {
        double r = angleModel(ts[i]) - thetas[i];
        residuals.push_back(r);
        sse += r*r;
    }
    double rmse = std::sqrt(sse / N);
    //打印初始值
    std::cout << "[初值] c=" << c_init << " b=" << b_init
              << " A=" << A_init << " Ω=" << bestOmega 
              << std::endl;
    //求解状态与代价
    std::cout << "  θ (初始角)     = " << theta0 << " rad\n";
    std::cout << "  b  (平均角速度）= " << b      << " rad/s\n";
    std::cout << "  A  (振幅)      = " << A      << " rad/s\n";
    std::cout << "  Ω  (频率参数)  = " << Omega  << " rad/s\n";
    std::cout << "  φ  (初相位)    = " << phi    << " rad\n";
    std::cout << "角度跨度 = " << (thetas.back() - thetas.front()) << " rad ("
              << ((thetas.back() - thetas.front()) / (2*PI)) << " 圈)\n";   // 量化指标
    std::cout << "有效样本数 = " << N << std::endl;   // 样本与帧范围
    std::cout << (summary.IsSolutionUsable() ? "解可用" : "解不可用") << std::endl;

    //建立画布
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(255,255,255));
    //观测值范围
    double ymin = 1e10, ymax = -1e10
         , xmin = ts.front(), xmax = ts.back();
    for (double v : thetas) { 
        ymin = std::min(ymin, v); 
        ymax = std::max(ymax, v); 
    }
    //y拟合后值的范围
    for (int j = 0; j <= 1000; ++j) {
        double t = xmin + (xmax - xmin) * j / 1000.0;
        double y = theta0 + b * t - (A / Omega) * std::cos(Omega * t + phi);
        ymin = std::min(ymin, y);
        ymax = std::max(ymax, y);
    }
    //为Y留边界
    double pad = 0.05 * (ymax - ymin);
    ymin -= pad;
    ymax += pad;
    Range range{xmin, xmax, ymin, ymax};
   
    //画拟合曲线
    int prev_px = 0,prev_py = 0;
    for (int j = 0; j <= 800; ++j) {
        double t = xmin + (xmax - xmin) * j / 800.0;
        double y = theta0 + b * t - (A / Omega) * std::cos(Omega * t + phi); // 复用模型
        int px, py;
        toPixel(t, y, range, px, py);
        if (j > 0){
            cv::line(canvas, cv::Point(prev_px, prev_py), cv::Point(px, py),
                     cv::Scalar(255, 0, 0), 2);              // 蓝色曲线
        }
        prev_px = px; prev_py = py;                          // 首点不连线，之后逐段连
    }
    //化观测点
    int step = std::max(1, (int)ts.size() / 144);
    for (size_t i = 0; i < ts.size(); i+=step*4){
        int px, py;
        toPixel(ts[i], thetas[i], range, px, py);
        cv::circle(canvas, cv::Point(px, py), 4,
                   cv::Scalar(0,0,255), -1);
    }
    //画坐标轴和标签
    drawxy(canvas);
    cv::putText(canvas, "t(s)",
                 cv::Point(W - margin -60, H - margin + 30),
                 cv::FONT_HERSHEY_SIMPLEX, 0.7,
                 cv::Scalar(0,0,0),2);
    cv::putText(canvas, "theta(rad)",
                 cv::Point(margin - 50, margin - 30),
                 cv::FONT_HERSHEY_SIMPLEX, 0.7,
                 cv::Scalar(0,0,0),2);
    // 图例(AI,可删)
    cv::circle(canvas, cv::Point(W - 200, 40), 5, cv::Scalar(0, 0, 255), -1);
    cv::putText(canvas, "observed", cv::Point(W - 185, 46),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    cv::line(canvas, cv::Point(W - 200, 70), cv::Point(W - 170, 70),
             cv::Scalar(255, 0, 0), 2);
    cv::putText(canvas, "fitted", cv::Point(W - 160, 76),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);//拟合曲线结束
    //-----------------------画残差图----------------------------
    //算残差
    double* p = params;//指针
    std::vector<double> res(ts.size());
    for (size_t i = 0; i < ts.size(); ++i) {
        AngleResidual functor(ts[i], thetas[i]);    // 用该点的 (t, θ) 构造
        functor.operator()<double>(p, &res[i]);     // 显式 <double>，残差写入 res[i]
    }
    //y的范围
    double rmin = 1e10, rmax = -1e10;
    for (double v : res){
        rmin = std::min(rmin,v);
        rmax = std::max(rmax,v);
    }
    rmin = std::min(rmin, 0.0);
    rmax = std::max(rmax, 0.0);
    double pad_r = 0.1 * (rmax - rmin);
    rmin -= pad_r;
    rmax += pad_r;
    //界定最小范围
    if (rmax - rmin < 1e-6) {
        rmin -= 1e-3;
        rmax += 1e-3;
    }

    Range range_2{ts.front(), ts.back(),rmin ,rmax };
    //画布
    cv::Mat canvas_r(H, W, CV_8UC3, cv::Scalar(255,255,255));
    //坐标轴
    drawxy(canvas_r);
    
    //0 基准线（灰色虚线）
    int zx1, zy0, zx2;
    toPixel(range_2.xmin, 0.0, range_2, zx1, zy0);
    toPixel(range_2.xmax, 0.0, range_2, zx2, zy0);
    for (int x = zx1; x < zx2; x += 12) {
        cv::line(canvas_r, cv::Point(x, zy0),
                 cv::Point(std::min(x + 6, zx2), zy0),
                 cv::Scalar(128, 128, 128), 1);
    }

    // 画残差点
    for (size_t i = 0; i < ts.size(); i+=step) {
        int px, py;
        toPixel(ts[i], res[i], range_2, px, py);
        cv::circle(canvas_r, cv::Point(px, py), 5, cv::Scalar(0,0,255), -1);
    }

    // 标签
    cv::putText(canvas_r, "residual (rad)",
                cv::Point(margin - 50, margin - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
    cv::putText(canvas_r, "t (s)",
                cv::Point(W - margin - 60, H - margin + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
    cv::putText(canvas_r, "0", cv::Point(margin - 25, zy0 + 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(128,128,128), 1);

    // 打印统计量
    /*double sse = 0, mean = 0;
    for (double r : res) { sse += r * r; mean += r; }
    mean /= res.size();
    double rmse = std::sqrt(sse / res.size());
    std::cout << "residual mean = " << mean << "\n";*/
    std::cout << "SSE = " << sse << ", RMSE = " << rmse << "\n";//残差图结束
    //---------------------------角速度曲线------------------------------------
    cv::Mat canvas_o (H, W, CV_8UC3, cv::Scalar(255,255,255));
    drawxy(canvas_o);

    double omegaMin = b - A;
    double omegaMax = b + A;
    double pad_o = 0.05 * (omegaMax - omegaMin);   // 5% 边距
    omegaMin -= pad_o;
    omegaMax += pad_o;

    const int SAMPLES = 800;
    // 先算第一个点
    Range range_o {xmin,xmax,omegaMin, omegaMax};
    double t0v = xmin;
    double w0  = b + A * std::sin(Omega * t0v + phi);

    toPixel(t0v, w0, range_o, prev_px, prev_py);   // 用 Range

    // 从第二个点开始，逐段连线
    for (int j = 1; j <= SAMPLES; ++j) {
        double t = xmin + (xmax - xmin) * j / static_cast<double>(SAMPLES);
        double w = b + A * std::sin(Omega * t + phi);

        int px, py;
        toPixel(t, w, range_o, px, py);
        cv::line(canvas_o, cv::Point(prev_px, prev_py), cv::Point(px, py),
                cv::Scalar(0, 0, 255), 2);
        prev_px = px; prev_py = py;
    }

    //水平参考线
    int pLx, pLy, pRx, pRy;
    toPixel(xmin, b, range_o, pLx, pLy);
    toPixel(xmax, b, range_o, pRx, pRy);
    cv::line(canvas_o, cv::Point(pLx, pLy), cv::Point(pRx, pRy),
             cv::Scalar(150,150,150), 1);
    cv::putText(canvas_o, "b (mean)", cv::Point(margin + 10, pRy - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(120,120,120), 1);
    //标签
    // y 轴标签
    cv::putText(canvas_o, "omega (rad/s)", cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1);
    // x 轴标签
    cv::putText(canvas_o, "Time (s)", cv::Point(W/2 - 40, H - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1);
    //保存
    std::string outputDir = "result/task2_fit";
    cv::imwrite(outputDir + "/fit_comparison.png", canvas);//拟合图
    cv::imwrite(outputDir + "/residuals.png", canvas_r);//残差图
    cv::imwrite(outputDir + "/angular_velocity.png", canvas_o);//角速度拟合图

    //------------------------视频标记--------------------------------------------------
    cv::VideoCapture cap2(inputPath);
    if (!cap2.isOpened()){
        std::cout<<"读取失败"<<std::endl;
        return -1;
    }
    const std::string videoOut = "result/task2_fit/tracking_overlay.mp4";
    //避免素材被覆盖
    if (videoOut == inputPath) {                       // ← 新增
        std::cerr << "错误：输出路径与输入路径相同，会覆盖素材！" << std::endl;
        return -1;
    }
    //文件格式
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(videoOut, fourcc, fps, cv::Size(width, height));

    if (!writer.isOpened()) {
        std::cout << "无法创建" << std::endl;
        return -1;
    }

    cv::Mat f;
    int idx2 = 0;
    while (cap2.read(f)) {
        cv::Point2f c;
        const cv::Point centerPx(480, 360);      // 旋转中心（int 点）

        // ── 1. 参考圆（最底层，灰色细线）──
        cv::circle(f, centerPx, 220, cv::Scalar(128,128,128), 1);

        // ── 2. 参数文字（顶部）──
        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      "theta0=%.3f b=%.4f A=%.4f Omega=%.4f phi=%.3f RMSE=%.4f",
                      theta0, b, A, Omega, phi, rmse);
        cv::putText(f, buf, cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,255,255), 1);

        // ── 3. 检测 + 标记 ──
        if (detectCyanCenter(f, c)) {
            // 旋转中心（白实心）
            cv::circle(f, centerPx, 6, cv::Scalar(255,255,255), cv::FILLED);
            // 目标中心（青实心）
            cv::circle(f, c, 8, cv::Scalar(0,255,255), cv::FILLED);
            // 两中心连线
            cv::line(f, centerPx, c, cv::Scalar(0,255,255), 2);

            // 当前角度
            double curTheta = std::atan2(cy - c.y, c.x - cx);
            std::snprintf(buf, sizeof(buf), "theta = %.3f rad", curTheta);
            cv::putText(f, buf, cv::Point(20, 90),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 2);
        } else {
            cv::putText(f, "NO DETECTION", cv::Point(20, 90),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
        }

        writer.write(f);
        ++idx2;
    }
    cap2.release();
    writer.release();
    return 0;
}//主函数结束

