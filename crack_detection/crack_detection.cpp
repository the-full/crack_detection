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
    // ��ȡͼ��  
    // TODO ��Ϊ����һ���ļ����µ������ļ�
    // ��������ԭͼ����Ȼ�����һ��ͬ·���µĹ̶����Ƶ��ļ���
    // EXTRA �ܹ��������ͼƬ����
    Mat img = imread(argv[1], IMREAD_COLOR);
    if (img.empty()) {
        cout << "�޷���ȡͼ��" << endl;
        return -1;
    }

    Mat inputImg;
    // ����ɫͼ��ת��Ϊ�Ҷ�ͼ��  
    cvtColor(img, inputImg, COLOR_BGR2GRAY);
    grayStretch(inputImg);
    //showTemp(inputImg);

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
    unevenLightCompensate(inputImg, 30);
    showTemp(inputImg);

    // ̬ͬ�˲�
    HomoTransform(inputImg, 3);
    showTemp(inputImg);


    //// ��ñ�任У�������ȹ���Ӱ��
    //// ��������±ȵ����˲����ã���Ӧ�أ���������±Ȳ���������
    //auto kernel2 = getStructuringElement(MORPH_RECT, Size(5, 5));
    //cv::morphologyEx(inputImg, inputImg, MORPH_BLACKHAT, kernel2);
    //grayReverse(inputImg);
    //showTemp(inputImg);

    // TODO �������������������̶��ļ���
    imwrite(String(argv[1]) + "tmep.jpg", inputImg);

    // ��ѹ��һ��ͼ�񣬴洢�ߴ�����֮��ָ�ͼ���С
    //auto oldRows = inputImg.rows;
    //auto oldCols = inputImg.cols;
    //auto scale = oldRows / 600.0;
    //cv::resize(inputImg, inputImg, { static_cast<int>(ceil(oldCols / scale)), 600 }, 0, 0, INTER_AREA);

    // TODO: �ϲ�
    //percolation_accelerate(inputImg);
    //denoise(inputImg);
    //percolation_accelerate(inputImg);
    percolation_overlap(inputImg);

    // �������ؽ���ͨ��
    //auto kernel3 = getStructuringElement(MORPH_RECT, Size(5, 5));
    //morphologyEx(inputImg, inputImg, MORPH_CLOSE, kernel3);

    auto outputImg = inputImg.clone();

    // �ָ�֮ǰ�Ĵ�С
    //resize(outputImg, outputImg, { oldCols, oldRows }, 0, 0, INTER_CUBIC);
    imshow("result", outputImg);
    waitKey(0);
    imwrite(String(argv[1]) + "result.jpg", outputImg);

    return 0;
}
