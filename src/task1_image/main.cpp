#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>    

//核
const int CORE = 15;

void step1_colorSpaces(const cv::Mat& img, const std::string& outputDir)
{
    // 转灰度图
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 转 HSV
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // 拆分 H / S / V 三个单通道
    cv::Mat hsvChannels[3];
    cv::split(hsv, hsvChannels);

    // 保存结果
    bool okGray = cv::imwrite(outputDir + "/01_gray.png", gray);
    bool okH    = cv::imwrite(outputDir + "/02_hsv_h.png",    hsvChannels[0]);
    bool okS    = cv::imwrite(outputDir + "/03_hsv_s.png",    hsvChannels[1]);
    bool okV    = cv::imwrite(outputDir + "/04_hsv_v.png",    hsvChannels[2]);

    /*std::cout << "[第1步] 尺寸: " << img.cols << " x " << img.rows << std::endl;
    std::cout << "[第1步] 灰度图 保存" << (okGray ? "成功" : "失败") << std::endl;
    std::cout << "[第1步] H 通道 保存" << (okH ? "成功" : "失败") << std::endl;
    std::cout << "[第1步] S 通道 保存" << (okS ? "成功" : "失败") << std::endl;
    std::cout << "[第1步] V 通道 保存" << (okV ? "成功" : "失败") << std::endl;*/
}
//滤波
void step2_filter(const cv::Mat& img, const std::string& outputDir)
{
    cv::Mat blurImg, gaussianImg, medianImg; 
    cv::blur(img, blurImg, cv::Size(CORE, CORE)); 
    cv::GaussianBlur(img, gaussianImg, cv::Size(CORE, CORE),0);  
    cv::medianBlur(img, medianImg, CORE); 
    //保存结果
    bool ok1 = cv::imwrite(outputDir + "/05_mean_filter.png",     blurImg);
    bool ok2 = cv::imwrite(outputDir + "/06_gaussian_filter.png", gaussianImg);
    bool ok3 = cv::imwrite(outputDir + "/07_median_filter.png",   medianImg);
}
//红色提取
void step3_extractred(const cv::Mat& img, const std::string& outputDir, cv::Mat& redMask)
{
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    //区间1
    cv::Mat mask1;
    cv::inRange(hsv, cv::Scalar(0, 100, 50),
                     cv::Scalar(10, 255, 255), mask1);
    //区间2
    cv::Mat mask2;
    cv::inRange(hsv, cv::Scalar(170, 100, 50),
                     cv::Scalar(179, 255, 255), mask2);
    //合并
    //cv::Mat redMask;
    cv::add(mask1, mask2, redMask);
    bool ok8 = cv::imwrite(outputDir + "/08_red_mask.png", redMask);

    int whitePix = cv::countNonZero(redMask);
    double ratio = 100.0 * whitePix / (redMask.rows * redMask.cols);

    //std::cout << "[第3步] 红色掩膜 保存" << (ok8 ? "成功" : "失败") << std::endl;
    std::cout << "阈值: H=0-10 或 170-179, S>=" << 100 << ", V>=" << 50 << std::endl;
    std::cout << "红色区域占比: " << ratio << "%" << std::endl;
}
//形态操作
void step4_shape(const cv::Mat& Img, const cv::Mat& img ,const std::string& outputDir)
{  
    const int K = 7;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(K,K));
    cv::Mat eroded, dilated, opened, closed;
    //四种操作
    cv::erode(img, eroded, kernel);
    cv::dilate(img, dilated, kernel);
    cv::morphologyEx(img, opened, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(img, closed, cv::MORPH_CLOSE, kernel);
    //保存图片
    cv::imwrite(outputDir + "/09_erode.png",  eroded);
    cv::imwrite(outputDir + "/10_dilate.png", dilated);
    cv::imwrite(outputDir + "/11_open.png",   opened);
    cv::imwrite(outputDir + "/12_close.png",  closed);
    //找轮廓
    std::vector<std::vector<cv::Point>>contours;
    cv::findContours(closed.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat canvas =Img.clone();
    const double AREA_THRESHOLD = 500;
    int drawn = 0, skipped = 0;

    for (size_t i = 0; i < contours.size(); ++i)
    {
        double area = cv::contourArea(contours[i]);
        //丢弃噪声
        if (area < AREA_THRESHOLD){ 
            skipped++;
            continue;
        }
        cv::drawContours(canvas, contours, (int)i, cv::Scalar(0, 0, 255), 2);

        //画外接矩形（绿色）
        // boundingRect 返回正矩形 Rect(x, y, width, height)
        cv::Rect box = cv::boundingRect(contours[i]);
        cv::rectangle(canvas, box, cv::Scalar(0, 255, 0), 2);
        //标注序号和面积
        char txt[64];
        snprintf(txt, sizeof(txt), "#%d A=%.0f", drawn, area);
        cv::putText(canvas, txt, cv::Point(box.x, box.y - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 0), 2);
        drawn++;   }

    cv::imwrite(outputDir + "/13_contours.png", canvas); 
    //第四步说明
    std::cout << "[第4步] 形态学四操作 + 轮廓筛选 完成" << std::endl;
    std::cout << "        kernel: MORPH_ELLIPSE " << K << "x" << K << std::endl;
    std::cout << "        面积阈值: " << AREA_THRESHOLD << std::endl;
    std::cout << "        保留轮廓 " << drawn << " 个, 丢弃 " << skipped << " 个" << std::endl;
}
//绘制与变换
void step5_draw_transform(const cv::Mat& img, const std::string& outputDir)
{
    // ---------- 5.1 绘制圆、矩形、文字 ----------
    cv::Mat canvas = img.clone();

    // 实心红圆。Scalar 是 BGR 顺序，(0,0,255)=红色。
    // 最后参数 -1 表示填充；写正数表示只画那么粗的圆环。
    cv::circle(canvas, cv::Point(200, 200), 80, cv::Scalar(0, 0, 255), -1);

    // 绿色空心矩形。Rect(左上角x, 左上角y, 宽, 高)
    // 注意不是 (x1,y1,x2,y2)！线宽 3 像素。
    cv::rectangle(canvas, cv::Rect(500, 150, 300, 200), cv::Scalar(0, 255, 0), 3);

    // 写文字。putText 不支持中文，只能英文/数字。
    // 第 5 个参数 2.0 是字体缩放系数（不是字号），大图要写大一点才看得清。
    cv::putText(canvas, "RoboMaster", cv::Point(60, 500),
                cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 255, 0), 3);

    cv::imwrite(outputDir + "/14_draw.png", canvas);

    // ---------- 5.2 绕图像中心旋转 35 度 ----------
    // getRotationMatrix2D 三个参数：旋转中心、旋转角度、缩放系数。
    // 角度为正表示逆时针（OpenCV 坐标系）。
    // 如果直接用 warpAffine 而不指定中心，是绕左上角原点转，那不是我们要的。
    cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, 35.0, 1.0);

    cv::Mat rotated;
    // 输出尺寸仍用 img.size()，所以旋转后四个角会被裁掉（题目未要求保留全图）
    cv::warpAffine(img, rotated, rotMat, img.size());
    cv::imwrite(outputDir + "/15_rotate35.png", rotated);

    // ---------- 5.3 裁剪左上角 1/4 ----------
    // 原宽原高各取一半。Rect(x, y, width, height)
    cv::Rect roi(0, 0, img.cols / 2, img.rows / 2);
    cv::Mat cropped = img(roi);
    cv::imwrite(outputDir + "/16_crop_topleft.png", cropped);

    // ---------- 说明输出 ----------
    std::cout << "[第5步] 绘制 + 变换 完成" << std::endl;
    std::cout << "        圆: 圆心(200,200) 半径80 实心红" << std::endl;
    std::cout << "        矩形: Rect(500,150,300,200) 绿框" << std::endl;
    std::cout << "        文字: RoboMaster 位置(60,500) 缩放2.0" << std::endl;
    std::cout << "        旋转: 绕中心(" << center.x << "," << center.y << ") 35度 缩放1.0" << std::endl;
    std::cout << "        裁剪: " << roi.width << "x" << roi.height
              << "（原图 " << img.cols << "x" << img.rows << " 的左上1/4）" << std::endl;
}


int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cout << "用法: " << argv[0] << " <输入图片> <输出目录>" << std::endl;
        return -1;
    }
    std::string inputPath = argv[1];
    std::string outputDir = argv[2];
    //读图
    cv::Mat img = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "读图失败！路径: " << inputPath << std::endl;
        return -1;
    }
    std::cout << "读图成功: " << inputPath << std::endl;
    //五个任务
    cv::Mat redMask;
    step1_colorSpaces(img, outputDir);
    step2_filter(img, outputDir);
    step3_extractred(img, outputDir, redMask);
    step4_shape(img, redMask, outputDir);
    step5_draw_transform(img, outputDir);
    cv::imwrite(outputDir + "/00_original.png", img);

    std::cout << "全部完成，结果已保存到: " << outputDir << std::endl;
    return 0;
}
