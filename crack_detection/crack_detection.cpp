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
    // ��ȡͼ��  
    Mat img = imread(argv[1], IMREAD_COLOR);
    if (img.empty()) {
        cout << "�޷���ȡͼ��" << endl;
        return -1;
    }

    Mat inputImg;
    // ����ɫͼ��ת��Ϊ�Ҷ�ͼ��  
    cvtColor(img, inputImg, COLOR_BGR2GRAY);
    grayStretch(inputImg);
    showTemp(inputImg);

    // ٤��任��ǿ�Աȶ�
    //GammaTransform(inputImg, 5);
    //showTemp(inputImg);

    // ���ղ���
    //unevenLightCompensate(inputImg, 32);
    //showTemp(inputImg);

    // �����˲�
    Mat tempImg;
    inputImg.convertTo(tempImg, CV_32FC1, 1.0 / 255.0);
    Mat cloneImg = tempImg.clone();
    guidedFilter(cloneImg, tempImg, 5, 0.01);
    tempImg.convertTo(inputImg, CV_8UC1, 255);
    showTemp(inputImg);

    // ���ղ���
    //unevenLightCompensate(inputImg, 32);
    //showTemp(inputImg);

    // ���ղ�����У��
    HomoTransform(inputImg, 3);
    showTemp(inputImg);


    // ��ֵ�˲�ȥ����������
    //medianBlur(inputImg, inputImg, (3, 3));
    //showTemp(inputImg);

    // ˫���˲���һ����ǿ�Աȶ�
    //Mat transImg;
    //bilateralFilter(inputImg, transImg, 5, 5, 5);
    //inputImg = std::move(transImg);
    //showTemp(inputImg);

    // ͼ��ƽ������
    //GaussianBlur(inputImg, inputImg, Size(5, 5), 9, 9);
    //showTemp(inputImg);
    //auto kernel1 = getStructuringElement(MORPH_RECT, Size(3, 3));
    //cv::morphologyEx(inputImg, inputImg, MORPH_OPEN, kernel1);
    //showTemp(inputImg);

    // ��ñ�任У�������ȹ���Ӱ��
    //auto kernel2 = getStructuringElement(MORPH_RECT, Size(5, 5));
    //cv::morphologyEx(inputImg, inputImg, MORPH_BLACKHAT, kernel2);
    //grayStretch(inputImg);
    //grayReverse(inputImg);
    //showTemp(inputImg);

    imwrite(String(argv[1]) + "tmep.jpg", inputImg);

    // ��ѹ��һ��ͼ�񣬴洢�ߴ�����֮��ָ�ͼ���С
    auto oldRows = inputImg.rows;
    auto oldCols = inputImg.cols;
    auto scale = oldRows / 600.0;
    //cv::resize(inputImg, inputImg, { static_cast<int>(ceil(oldCols / scale)), 600 }, 0, 0, INTER_AREA);

    // TODO: �ϲ�
    percolation(inputImg);
    denoise(inputImg);

    // �������ؽ���ͨ��
    //auto kernel3 = getStructuringElement(MORPH_RECT, Size(5, 5));
    //morphologyEx(inputImg, inputImg, MORPH_CLOSE, kernel3);

    auto outputImg = inputImg.clone();

    // �ָ�֮ǰ�Ĵ�С
    resize(outputImg, outputImg, { oldCols, oldRows }, 0, 0, INTER_CUBIC);
    imshow("result", outputImg);
    waitKey(0);
    imwrite(String(argv[1]) + "result.jpg", outputImg);

    return 0;
}
