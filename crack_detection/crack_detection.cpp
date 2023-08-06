#include "opencv2/core.hpp"
#include "opencv2/core/base.hpp"
#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <unordered_set>
#include <map>
#include <assert.h>
#include <algorithm>

#pragma comment(lib, "./opencv_world480.lib")

// Let's try C++20, make C++ great again
import preprocessing;
import Corr;
import percolation;

using namespace std;
using namespace cv;


#define DEBUG
void showTemp(Mat& img, String title = "temp") {
#ifdef DEBUG
    imshow(title, img);
    waitKey(0);
#endif
}

inline auto showRegion(const Mat& img, const auto& Dp, const auto& center, double Fc = 0, int time=0) {
    auto midImg = img.clone();
    for (const auto& p : Dp) {
        auto corr = p.second;
        midImg.at<uchar>(corr.x, corr.y) = 0;
    }
    putText(midImg, "Fc=" + to_string(Fc), { 50, 50 }, 5, 1, 1);
    imshow("mid result", midImg);
    waitKey(time);
    return Fc;
}


int main(int argc, char** argv) {
    // 读取图像  
    Mat img = imread(argv[1], IMREAD_COLOR);
    if (img.empty()) {
        cout << "无法读取图像" << endl;
        return -1;
    }

    Mat inputImg;
    // 将彩色图像转换为灰度图像  
    cvtColor(img, inputImg, COLOR_BGR2GRAY);
    grayStretch(inputImg);
    showTemp(inputImg);

    // 伽马变换增强对比度
    //GammaTransform(inputImg, 5);
    //showTemp(inputImg);

    // 光照补偿
    //unevenLightCompensate(inputImg, 32);
    //showTemp(inputImg);

    // 导向滤波
    Mat tempImg;
    inputImg.convertTo(tempImg, CV_32FC1, 1.0 / 255.0);
    Mat cloneImg = tempImg.clone();
    guidedFilter(cloneImg, tempImg, 5, 0.01);
    tempImg.convertTo(inputImg, CV_8UC1, 255);
    showTemp(inputImg);

    // 光照补偿
    //unevenLightCompensate(inputImg, 32);
    //showTemp(inputImg);

    // 光照不均匀校正
    HomoTransform(inputImg, 3);
    showTemp(inputImg);


    // 中值滤波去除椒盐噪声
    //medianBlur(inputImg, inputImg, (3, 3));
    //showTemp(inputImg);

    // 双边滤波进一步增强对比度
    //Mat transImg;
    //bilateralFilter(inputImg, transImg, 5, 5, 5);
    //inputImg = std::move(transImg);
    //showTemp(inputImg);

    // 图像平滑处理
    //GaussianBlur(inputImg, inputImg, Size(5, 5), 9, 9);
    //showTemp(inputImg);
    //auto kernel1 = getStructuringElement(MORPH_RECT, Size(3, 3));
    //cv::morphologyEx(inputImg, inputImg, MORPH_OPEN, kernel1);
    //showTemp(inputImg);

    // 低帽变换校正不均匀光照影响
    //auto kernel2 = getStructuringElement(MORPH_RECT, Size(5, 5));
    //cv::morphologyEx(inputImg, inputImg, MORPH_BLACKHAT, kernel2);
    //grayStretch(inputImg);
    //grayReverse(inputImg);
    //showTemp(inputImg);

    imwrite(String(argv[1]) + "tmep.jpg", inputImg);

    // 先压缩一下图像，存储尺寸用于之后恢复图像大小
    auto oldRows = inputImg.rows;
    auto oldCols = inputImg.cols;
    auto scale = oldRows / 600.0;
    //cv::resize(inputImg, inputImg, { static_cast<int>(ceil(oldCols / scale)), 600 }, 0, 0, INTER_AREA);

    // TODO: 合并
    percolation(inputImg);
    denoise(inputImg);

    // 闭运算重建连通性
    //auto kernel3 = getStructuringElement(MORPH_RECT, Size(5, 5));
    //morphologyEx(inputImg, inputImg, MORPH_CLOSE, kernel3);

    auto outputImg = inputImg.clone();

    // 恢复之前的大小
    resize(outputImg, outputImg, { oldCols, oldRows }, 0, 0, INTER_CUBIC);
    imshow("result", outputImg);
    waitKey(0);
    imwrite(String(argv[1]) + "result.jpg", outputImg);

    return 0;
}
