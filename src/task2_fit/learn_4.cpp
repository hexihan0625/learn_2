// ── 假设这里参数都算好了：theta0, b, A, Omega, phi, rmse ──

// 重新打开视频
cv::VideoCapture cap2(inputPath);
if (!cap2.isOpened()) {
    std::cout << "无法重新打开视频！" << std::endl;
    return -1;
}

// 创建 writer
const std::string videoOut = "result/task2_fit/tracking_overlay.mp4";
int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
cv::VideoWriter writer(videoOut, fourcc, fps, cv::Size(width, height));
if (!writer.isOpened()) {
    std::cout << "无法创建输出视频！" << std::endl;
    return -1;
}

// 逐帧画
cv::Mat f;
int idx2 = 0;
while (cap2.read(f)) {
    // 1. 检测目标
    cv::Point2f c;
    if (detectCyanCenter(f, c)) {
        // 画标记
    } else {
        // 画"未检测到"
    }
    // 2. 写参数文字
    // 3. 写入
    writer.write(f);
    ++idx2;
}

cap2.release();
writer.release();
