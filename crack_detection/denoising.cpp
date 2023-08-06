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
// M �� N ��Ϊ���ڴ�С
// ����Ҫ�������ĵ㣬�� M �� N Ϊ���� 
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
    // ������������
    Corr center(r, c);

    // ���� localWindow ���ұ�
    /* localWindow
     *  0x00  ==>  ���� Dp �У�Ҳ���� Dc ��
     *  0xf0  ==>  �� Dp ��
     *  0x0f  ==>  �� Dc �У���δ���� Dp
     *  0xff  ==>  ���� Dc ��, ���� Dp ��
     * ֻ�е�һ������¿��ܽ������ Dc
     */
    array< array<uchar, M>, M> localWindow{};
    // localWindow �������
    auto luPoint = center - Corr{RM, RM};

    // ��ʼ�� Dp �� Dc
    set<decltype(center)> Dp{}, Dc{};

    Dp.insert(center);

    while (true) {
        // ���� Dp����ÿ�����ص㣬����������򣬸��� Dc
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
        // ��û�п����������ص��� upIter == Dc.begin()
        // ��ʱѡ�� Dc.begin() ��Ϊ������
        if (upIter == Dc.begin())
            upIter++;
        Dp.insert(Dc.begin(), upIter);
        for (auto lt = Dc.begin(); lt != upIter; lt++) {
            // ������ص��״̬
            auto disp = lt->second - luPoint;
            localWindow[disp.x][disp.y] |= 0xf0;
            // ͬʱ����Ƿ񳬳�����
            auto vec = lt->second - center;
            if (abs(vec.x) > RN || abs(vec.y) > RN) {
                isOutofWindow = true;
            }
        }
        Dc.erase(Dc.begin(), upIter);
    }
    // ���� N
    RN++; N += 2;
    isOutofWindow = false;
    while (N < M) {
        // ���� T
        if (T != 255)
            T = max(Dp.begin()->first, T) + w;

        // ���� Dp����ÿ�����ص㣬����������򣬸��� Dc
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
        // �� Dp �޷����£���ֹ����
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
        // ���򣬼������� Dp
        Dp.insert(Dc.begin(), upIter);
        for (auto lt = Dc.begin(); lt != upIter; lt++) {
            // ������ص��״̬
            auto disp = lt->second - luPoint;
            localWindow[disp.x][disp.y] |= 0xf0;
            // ͬʱ����Ƿ񳬳�����
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
    // ������������
    Corr center(r, c);

    // ���� localWindow ���ұ�
    /* localWindow
     *  0x00  ==>  ���� Dp �У�Ҳ���� Dc ��
     *  0xf0  ==>  �� Dp ��
     *  0x0f  ==>  �� Dc �У���δ���� Dp
     *  0xff  ==>  ���� Dc ��, ���� Dp ��
     * ֻ�е�һ������¿��ܽ������ Dc
     */
    array< array<uchar, M>, M> localWindow{};
    // localWindow �������
    auto luPoint = center - Corr{RM, RM};

    // ��ʼ�� Dp �� Dc
    auto T = img.at<uchar>(r, c);
    pair<uchar, Corr> center_point(T, center);
    set<decltype(center_point)> Dp{}, Dc{};
    Dp.insert(center_point);

    while (true) {
        // ���� Dp����ÿ�����ص㣬����������򣬸��� Dc
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
                    // ��ǵ�Ϊ������
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
            // ������ص��״̬
            auto disp = lt->second - luPoint;
            localWindow[disp.x][disp.y] |= 0xf0;
        }
        Dc.erase(Dc.begin(), upIter);
    }
}

int main(int argc, char** argv) {
    // ��ȡͼ��  
    Mat inputImg = imread(argv[1], IMREAD_GRAYSCALE);

    // ��ȡ�����������Ա���ͼ��
    auto rows = inputImg.rows;
    auto cols = inputImg.cols;

    // ���ͼ����ʡȥԽ����
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
    // ȥ��֮ǰ�����
    outputImg = outputImg(Rect(RM, RM, cols, rows));
    // �ָ�֮ǰ�Ĵ�С
    resize(outputImg, outputImg, { oldCols, oldRows }, 0, 0, INTER_CUBIC);
    imshow("result image", outputImg);
    waitKey(0);
    imwrite(String(argv[1]) + "result.jpg", outputImg);

    return 0;
}
