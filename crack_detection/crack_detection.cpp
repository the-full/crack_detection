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
import DEBUG;

using namespace std;
using namespace cv;


int main(int argc, char** argv) {
    // 读取图像  
    // TODO 改为处理一个文件夹下的所有文件
    // 处理结果与原图叠放然后存入一个同路径下的固定名称的文件夹
    // EXTRA 能够处理多张图片输入
    Mat img = imread(argv[1], IMREAD_COLOR);
    if (img.empty()) {
        cout << "无法读取图像" << endl;
        return -1;
    }

    Mat inputImg;
    // 将彩色图像转换为灰度图像  
    cvtColor(img, inputImg, COLOR_BGR2GRAY);
    grayStretch(inputImg);
    //showTemp(inputImg);

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
    unevenLightCompensate(inputImg, 30);
    showTemp(inputImg);

    // 同态滤波
    HomoTransform(inputImg, 3);
    showTemp(inputImg);


    //// 低帽变换校正不均匀光照影响
    //// 部分情况下比导向滤波更好，相应地，部分情况下比不做处理还差
    //auto kernel2 = getStructuringElement(MORPH_RECT, Size(5, 5));
    //cv::morphologyEx(inputImg, inputImg, MORPH_BLACKHAT, kernel2);
    //grayReverse(inputImg);
    //showTemp(inputImg);

    // TODO 更合理的命名，并放入固定文件夹
    imwrite(String(argv[1]) + "tmep.jpg", inputImg);

    // 先压缩一下图像，存储尺寸用于之后恢复图像大小
    //auto oldRows = inputImg.rows;
    //auto oldCols = inputImg.cols;
    //auto scale = oldRows / 600.0;
    //cv::resize(inputImg, inputImg, { static_cast<int>(ceil(oldCols / scale)), 600 }, 0, 0, INTER_AREA);

    // TODO: 合并
    //percolation_accelerate(inputImg);
    //denoise(inputImg);
    //percolation_accelerate(inputImg);
    percolation_overlap(inputImg);

    // 闭运算重建连通性
    //auto kernel3 = getStructuringElement(MORPH_RECT, Size(5, 5));
    //morphologyEx(inputImg, inputImg, MORPH_CLOSE, kernel3);

    auto outputImg = inputImg.clone();

    // 恢复之前的大小
    //resize(outputImg, outputImg, { oldCols, oldRows }, 0, 0, INTER_CUBIC);
    imshow("result", outputImg);
    waitKey(0);
    imwrite(String(argv[1]) + "result.jpg", outputImg);

    return 0;
}
