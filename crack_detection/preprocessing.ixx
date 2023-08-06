module;

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <array>

export module preprocessing;

using namespace cv;


export void grayStretch(Mat& img) noexcept {
    auto rows = img.rows;
    auto cols = img.cols;
    double minVal, maxVal;
    minMaxLoc(img, &minVal, &maxVal);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            img.at<uchar>(r, c) = (img.at<uchar>(r, c) - minVal) * 255.0 / (maxVal - minVal);
        }
    }
}

export void grayReverse(Mat& img) noexcept {
    auto rows = img.rows;
    auto cols = img.cols;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // �ټ�һ�㣬������� 255 �׵��޷�������
            uchar val = 250 - img.at<uchar>(r, c);
            if (val < 0) val = 0;
            img.at<uchar>(r, c) = val;
        }
    }
}

export void GammaTransform(Mat& img, double gamma)
{
    std::array<uchar, 256> lut{};
    for (int i = 0; i < 256; i++)
    {
        lut[i] = saturate_cast<uchar>(pow((float)i / 255.0, gamma) * 255.0f);
    }
    auto rows = img.rows;
    auto cols = img.cols;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            img.at<uchar>(r, c) = lut[img.at<uchar>(r, c)];
        }
    }
}

export void HomoTransform(Mat& img, int sigma)
{
    Mat src = img.clone();
    Mat doubleImage, gaussianImage, logIImage, logGImage, logRImage;
    //ת����Χ������ͼ��Ԫ������1.0��֤log��������,��ֹ���
    src.convertTo(doubleImage, CV_64FC1, 1.0, 1.0);

    //��˹ģ������sizeΪ��ʱ��ͨ��sigma�Զ����м���
    GaussianBlur(doubleImage, gaussianImage, Size(0, 0), sigma);

    //OpenCV��log�������Լ��������ֵ��logIImage��logGImage���Ƕ�������Ľ����
    log(doubleImage, logIImage);
    log(gaussianImage, logGImage);

    logRImage = logIImage - logGImage;
    exp(logRImage, doubleImage);
    normalize(doubleImage, img, 0, 255, NORM_MINMAX, CV_8UC1);
}

export void guidedFilter(const Mat& src, Mat& dst, int radius, double eps)
{
    Mat I = src;
    Mat p = src;
    int d = 2 * radius + 1;

    Mat N;
    boxFilter(Mat::ones(src.size(), CV_32F), N, CV_32F, Size(d, d));

    // �����ֵ
    Mat mean_I;
    boxFilter(I, mean_I, CV_32F, Size(d, d));
    Mat mean_p;
    boxFilter(p, mean_p, CV_32F, Size(d, d));

    // ���� I * I, p * I
    Mat mean_I2;
    boxFilter(I.mul(I), mean_I2, CV_32F, Size(d, d));
    Mat mean_Ip;
    boxFilter(I.mul(p), mean_Ip, CV_32F, Size(d, d));

    // ���㷽���Э����
    Mat var_I = mean_I2 - mean_I.mul(mean_I);
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

    // ���� a �� b
    Mat a = cov_Ip / (var_I + eps);
    Mat b = mean_p - a.mul(mean_I);

    // �����ֵ
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_32F, Size(d, d));
    boxFilter(b, mean_b, CV_32F, Size(d, d));

    // �������ͼ��
    dst = mean_a.mul(I) + mean_b;
}

export void unevenLightCompensate(Mat& image, int blockSize)
{
    if (image.channels() == 3) cvtColor(image, image, 7);
    double average = mean(image)[0];
    int rows_new = ceil(double(image.rows) / double(blockSize));
    int cols_new = ceil(double(image.cols) / double(blockSize));
    Mat blockImage;
    blockImage = Mat::zeros(rows_new, cols_new, CV_32FC1);
    for (int i = 0; i < rows_new; i++)
    {
        for (int j = 0; j < cols_new; j++)
        {
            int rowmin = i * blockSize;
            int rowmax = (i + 1) * blockSize;
            if (rowmax > image.rows) rowmax = image.rows;
            int colmin = j * blockSize;
            int colmax = (j + 1) * blockSize;
            if (colmax > image.cols) colmax = image.cols;
            Mat imageROI = image(Range(rowmin, rowmax), Range(colmin, colmax));
            double temaver = mean(imageROI)[0];
            blockImage.at<float>(i, j) = temaver;
        }
    }
    blockImage = blockImage - average;
    Mat blockImage2;
    resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), INTER_CUBIC);
    Mat image2;
    image.convertTo(image2, CV_32FC1);
    Mat dst = image2 - blockImage2;
    dst.convertTo(image, CV_8UC1);
}

export auto highPass(cv::Mat& img) {
    auto img_temp = img.clone();
    cv::GaussianBlur(img, img_temp, cv::Size(5, 5), 3, 3);
    auto img_temp2 = img - img_temp;
    cv::addWeighted(img_temp2, 1, img, 1, 0, img);
}