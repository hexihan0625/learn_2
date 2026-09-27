    // ==================== 图 3：真实空间 X-Y 轨迹 ====================
    // ---- 从角度还原 x-y 位置 ----
    auto toXY = [&](double theta, double R, double& x, double& y) {
        x = cx + R * std::cos(theta);
        y = cy - R * std::sin(theta);     // 图像 y 轴向下，用减号
    };

    const double R = 200.0;               // ★ 小球到中心的距离（按你的视频调整）

    // ---- 生成拟合轨迹（用拟合参数，均匀采样）----
    std::vector<double> fitX, fitY;
    for (int j = 0; j <= 720; ++j) {
        double t = xmin + (xmax - xmin) * j / 720.0;
        double th = theta0 + b * t - (A / Omega) * std::cos(Omega * t + phi);  // 拟合角度
        double x, y;
        toXY(th, R, x, y);
        fitX.push_back(x);
        fitY.push_back(y);
    }

    // ---- 生成观测轨迹（用观测时刻和角度）----
    std::vector<double> obsX, obsY;
    for (size_t i = 0; i < ts.size(); ++i) {
        double x, y;
        toXY(thetas[i], R, x, y);
        obsX.push_back(x);
        obsY.push_back(y);
    }

    // ---- 确定 x/y 范围（同时包含观测和拟合，且关于中心对称更美观）----
    double xmin3 = 1e18, xmax3 = -1e18, ymin3 = 1e18, ymax3 = -1e18;
    for (double v : obsX) { xmin3 = std::min(xmin3, v); xmax3 = std::max(xmax3, v); }
    for (double v : fitX) { xmin3 = std::min(xmin3, v); xmax3 = std::max(xmax3, v); }
    for (double v : obsY) { ymin3 = std::min(ymin3, v); ymax3 = std::max(ymax3, v); }
    for (double v : fitY) { ymin3 = std::min(ymin3, v); ymax3 = std::max(ymax3, v); }

    // 正圆/对称运动：让 x、y 用同一个范围，避免圆被压成椭圆
    double cxmid = 0.5 * (xmin3 + xmax3);
    double cymid = 0.5 * (ymin3 + ymax3);
    double rad = 0.5 * std::max(xmax3 - xmin3, ymax3 - ymin3);
    rad *= 1.08;                       // 留 8% 边距
    xmin3 = cxmid - rad; xmax3 = cxmid + rad;
    ymin3 = cymid - rad; ymax3 = cymid + rad;

    Range range3{xmin3, xmax3, ymin3, ymax3};

    // ---- 画布 ----
    cv::Mat canvas3(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

    // ---- 先画拟合轨迹（粗蓝线），后画观测点（红点压在上面）----
    for (size_t i = 1; i < fitX.size(); ++i) {
        int px0, py0, px1, py1;
        toPixel(fitX[i-1], fitY[i-1], range3, px0, py0);
        toPixel(fitX[i],   fitY[i],   range3, px1, py1);
        cv::line(canvas3, cv::Point(px0, py0), cv::Point(px1, py1),
                 cv::Scalar(255, 0, 0), 2);          // 蓝色拟合轨迹
    }

    int step3 = std::max(1, (int)ts.size() / 144);   // 观测点降采样到约 144
    for (size_t i = 0; i < obsX.size(); i += step3) {
        int px, py;
        toPixel(obsX[i], obsY[i], range3, px, py);
        cv::circle(canvas3, cv::Point(px, py), 4, cv::Scalar(0, 0, 255), -1);  // 红点
    }

    // ---- 坐标轴 + 标签 ----
    cv::line(canvas3, cv::Point(margin, H - margin),
             cv::Point(W - margin, H - margin), cv::Scalar(0,0,0), 1);   // x 轴
    cv::line(canvas3, cv::Point(margin, margin),
             cv::Point(margin, H - margin), cv::Scalar(0,0,0), 1);       // y 轴

    cv::putText(canvas3, "x (px)",
                cv::Point(W - margin - 70, H - margin + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
    cv::putText(canvas3, "y (px)",
                cv::Point(margin - 50, margin - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);

    // ---- 图例 ----
    cv::circle(canvas3, cv::Point(W - 200, 40), 5, cv::Scalar(0,0,255), -1);
    cv::putText(canvas3, "observed", cv::Point(W - 185, 46),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1);
    cv::line(canvas3, cv::Point(W - 200, 70), cv::Point(W - 170, 70),
             cv::Scalar(255,0,0), 2);
    cv::putText(canvas3, "fitted", cv::Point(W - 160, 76),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1);

    // ---- 保存 ----
    cv::imwrite(outputDir + "trajectory.png", canvas3);
