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

using namespace std;
using namespace cv;

constexpr double PI = 3.14159265;
constexpr double Threshold_FcMax = 0.7;
constexpr double Threshold_RadiusMax = 20;
// M 和 N 均为窗口大小
// 窗口要包含中心点，故 M 和 N 为奇数 
constexpr int RM = 40;
constexpr int M = 2 * RM + 1;

void grayStretch(Mat& img) noexcept {
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

struct Corr {
    int x, y;

    Corr() = default;
    Corr(const Corr& c) = default;
    Corr(Corr&& c) = default;
    ~Corr() = default;

    Corr(int x, int y) : x(x), y(y) {}


    inline bool operator<(const Corr& c) const {
        if (this->x != c.x)
            return this->x < c.x;
        else
            return this->y < c.y;
    }

    inline bool operator>(const Corr& c) const {
        if (this->x != c.x)
            return this->x > c.x;
        else
            return this->y > c.y;
    }

    inline bool operator==(const Corr& c) const {
        return (this->x == c.x && this->y == c.y);
    }

    inline Corr operator+(const Corr& c) const {
        return Corr(this->x + c.x, this->y + c.y);
    }

    inline Corr operator-(const Corr& c) const {
        return Corr(this->x - c.x, this->y - c.y);
    }
};

inline auto corr_norm(const auto& c) noexcept {
    return sqrt(c.x * c.x + c.y * c.y);
}

const Corr directionTable[8] = {
    {-1, -1}, {-1, 0}, {-1, 1},
    { 0, -1}, { 0, 1},
    { 1, -1}, { 1, 0},  {1, 1},
};

auto showRegion(const Mat& img, const auto& Dp, const auto& center) {
    //auto midImg = img.clone();
    auto Dp_max = 0;
    auto Fc = (Dp.size()) / (M * 1.0 * M);
    //putText(midImg, "Fc=" + to_string(Fc) + "R=" + to_string(Dp_max), { 50, 50 }, 5, 1, 1);
    //imshow("mid result", midImg);
    //waitKey(0);
    return Fc;
}

auto denoise(const Mat& img, const auto& cente) {
    // 中心像素坐标
    Corr center(r, c);

    // 设置 localWindow 查找表
    /* localWindow
     *  0x00  ==>  不在 Dp 中，也不在 Dc 中
     *  0xf0  ==>  在 Dp 中
     *  0x0f  ==>  在 Dc 中，还未加入 Dp
     *  0xff  ==>  曾在 Dc 中, 现在 Dp 中
     * 只有第一种情况下可能将点加入 Dc
     */
    array< array<uchar, M>, M> localWindow{};
    // localWindow 起点坐标
    auto luPoint = center - Corr{RM, RM};

    // 初始化 Dp 和 Dc
    set<decltype(center)> Dp{}, Dc{};

    Dp.insert(center);

    while (true) {
        // 遍历 Dp，对每个像素点，遍历其八邻域，更新 Dc
        for (const auto& p : Dp) {
            for (const auto& direction : directionTable) {
                auto neiborPoint = p.second + direction;
                auto disp = neiborPoint - luPoint;

                if ()
                if (localWindow[disp.x][disp.y] != 0x00) continue;

                auto val = img.at<uchar>(neiborPoint.x, neiborPoint.y);
                Dc.insert({ val, neiborPoint });
                localWindow[disp.x][disp.y] |= 0x0f;
            }
        }

        auto upIter = Dc.upper_bound({ T, {0, 0} });
        // 若没有可渗流的像素点则 upIter == Dc.begin()
        // 此时选择 Dc.begin() 做为渗流点
        if (upIter == Dc.begin())
            upIter++;
        Dp.insert(Dc.begin(), upIter);
        for (auto lt = Dc.begin(); lt != upIter; lt++) {
            // 更新相关点的状态
            auto disp = lt->second - luPoint;
            localWindow[disp.x][disp.y] |= 0xf0;
            // 同时检查是否超出区域
            auto vec = lt->second - center;
            if (abs(vec.x) > RN || abs(vec.y) > RN) {
                isOutofWindow = true;
            }
        }
        Dc.erase(Dc.begin(), upIter);
    }
    // 更新 N
    RN++; N += 2;
    isOutofWindow = false;
    while (N < M) {
        // 更新 T
        if (T != 255)
            T = max(Dp.begin()->first, T) + w;

        // 遍历 Dp，对每个像素点，遍历其八邻域，更新 Dc
        for (const auto& p : Dp) {
            for (const auto& direction : directionTable) {
                auto neiborPoint = p.second + direction;
                auto disp = neiborPoint - luPoint;

                if (localWindow[disp.x][disp.y] != 0x00) continue;

                auto val = img.at<uchar>(neiborPoint.x, neiborPoint.y);
                Dc.insert({ val, neiborPoint });
                localWindow[disp.x][disp.y] |= 0x0f;
            }
        }

        auto upIter = Dc.upper_bound({ T, {0, 0} });
        // 若 Dp 无法更新，终止渗流
        if (upIter == Dc.begin()) {
            auto Fc = showRegion(img, Dp, center);
            if (Fc < Threshold_Crack) {
                for (const auto& p : Dp) {
                    marks[p.second.x][p.second.y] = 1;
                    img.at<uchar>(p.second.x, p.second.y) = 255 * Fc;
                }
            }
            else if (Fc >= Threshold_Background) {
                for (const auto& p : Dp) {
                    int val = center_point.first + Diff_Background;
                    if (p.first > val) {
                        marks[p.second.x][p.second.y] = -1;
                    }
                }
            }
            return Fc;
        }
        // 否则，继续更新 Dp
        Dp.insert(Dc.begin(), upIter);
        for (auto lt = Dc.begin(); lt != upIter; lt++) {
            // 更新相关点的状态
            auto disp = lt->second - luPoint;
            localWindow[disp.x][disp.y] |= 0xf0;
            // 同时检查是否超出区域
            auto vec = lt->second - center;
            if (abs(vec.x) > RN || abs(vec.y) > RN) {
                isOutofWindow = true;
                break;
            }
        }
        Dc.erase(Dc.begin(), upIter);

        if (isOutofWindow) {
            RN++; N += 2;
            isOutofWindow = false;
        }
    }
    auto Fc = showRegion(img, Dp, center);
    if (Fc < Threshold_Crack) {
        for (const auto& p : Dp) {
            marks[p.second.x][p.second.y] = 1;
            img.at<uchar>(p.second.x, p.second.y) = 255 * Fc;
        }
    }
    else if (Fc > Threshold_Background) {
        for (const auto& p : Dp) {
            int val = center_point.first + Diff_Background;
            if (p.first > val) {
                marks[p.second.x][p.second.y] = -1;
            }
        }
    }
    return Fc;
}

auto denoise(const Mat& img, int r, int c) {
    // 中心像素坐标
    Corr center(r, c);

    // 设置 localWindow 查找表
    /* localWindow
     *  0x00  ==>  不在 Dp 中，也不在 Dc 中
     *  0xf0  ==>  在 Dp 中
     *  0x0f  ==>  在 Dc 中，还未加入 Dp
     *  0xff  ==>  曾在 Dc 中, 现在 Dp 中
     * 只有第一种情况下可能将点加入 Dc
     */
    array< array<uchar, M>, M> localWindow{};
    // localWindow 起点坐标
    auto luPoint = center - Corr{RM, RM};

    // 初始化 Dp 和 Dc
    auto T = img.at<uchar>(r, c);
    pair<uchar, Corr> center_point(T, center);
    set<decltype(center_point)> Dp{}, Dc{};
    Dp.insert(center_point);

    while (true) {
        // 遍历 Dp，对每个像素点，遍历其八邻域，更新 Dc
        for (const auto& p : Dp) {
            for (const auto& direction : directionTable) {
                auto neiborPoint = p.second + direction;
                auto disp = neiborPoint - luPoint;

                if (disp.x < 0 || disp.x >= M || disp.y < 0 || disp.y >= M) continue;
                if (localWindow[disp.x][disp.y] != 0x00)                    continue;

                auto val = img.at<uchar>(neiborPoint.x, neiborPoint.y);
                Dc.insert({ val, neiborPoint });
                localWindow[disp.x][disp.y] |= 0x0f;
            }
        }

        auto upIter = Dc.upper_bound({ 255 * Threshold_FcMax, {0, 0} });
        if (upIter == Dc.begin()) {
            //auto midImg = img.clone();
            auto radius = 0.0;
            for (const auto& p : Dp) {
                radius = max(radius, corr_norm(p.second));
            }
            auto Fc = (Dp.size()) / (PI * radius * radius);
            if (Fc > Threshold_FcMax || radius < Threshold_RadiusMax) {
                for (const auto& p : Dp) {
                    // 标记点为背景点
                    marks[p.second.x][p.second.y] = -1;
                }
            }
            //putText(midImg, "Fc=" + to_string(Fc) + "R=" + to_string(Dp_max), { 50, 50 }, 5, 1, 1);
            //imshow("mid result", midImg);
            //waitKey(0);
            return;
        }
        Dp.insert(Dc.begin(), upIter);
        for (auto lt = Dc.begin(); lt != upIter; lt++) {
            // 更新相关点的状态
            auto disp = lt->second - luPoint;
            localWindow[disp.x][disp.y] |= 0xf0;
        }
        Dc.erase(Dc.begin(), upIter);
    }
}

int main(int argc, char** argv) {
    // 读取图像  
    Mat inputImg = imread(argv[1], IMREAD_GRAYSCALE);

    // 获取行数和列数以遍历图像
    auto rows = inputImg.rows;
    auto cols = inputImg.cols;

    // 填充图像以省去越界检查
    Mat outputImg = inputImg.clone();

    vector<vector<int>> marks(rows + M, vector<int>(cols + M, 0));
    for (int r = 0; r < rows; r++) {
        printf("current rows = %d\n", r - RM);
        for (int c = 0; c < cols; c++) {
            if (inputImg.at<uchar>(r, c) < 255 * Threshold_FcMax)
                auto isBackground = denoise(inputImg, r, c);
            }
            if (isBackground) = 
        }
    }
    // 去掉之前的填充
    outputImg = outputImg(Rect(RM, RM, cols, rows));
    // 恢复之前的大小
    resize(outputImg, outputImg, { oldCols, oldRows }, 0, 0, INTER_CUBIC);
    imshow("result image", outputImg);
    waitKey(0);
    imwrite(String(argv[1]) + "result.jpg", outputImg);

    return 0;
}
