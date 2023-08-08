export;

#include <opencv2/opencv.hpp>

export module DEBUG;

#define DEBUG
export void showTemp(cv::Mat& img, cv::String title = "temp") {
#ifdef DEBUG
    imshow(title, img);
    cv::waitKey(0);
#endif
}
