module;

#include <opencv2/opencv.hpp>  
#include <utility>
#include <iostream>

export module percolation;

using namespace std;
import Corr;

//// 常量
constexpr double PI = 3.14159265;
const Corr directionTable[8] = {
    {-1, -1}, {-1, 0}, {-1, 1},
    { 0, -1}, { 0, 1},
    { 1, -1}, { 1, 0},  {1, 1},
};


//// 算法参数 ==> 可以整成结构体，但现在没必要
/* 渗流算法的参数
 * Threshold                原算法判定的阈值，Fc < T 则判为裂缝，否则判为背景
 * Threshold_Crack          加速算法判定裂缝的阈值，Fc <  T 则认为渗流区域（Dp）都是裂缝
 * Threshold_Background     加速算法判定背景的阈值，Fc >= T 则认为渗流区域（Dp）可能是背景，取决于灰度的差值
 * Diff_Background          加速算法判定背景的差值，Val - Fc >= D 时判为背景
 * N, M, RN, RM             渗流最小（最大）窗口大小及其半径
 * 注：N 在计算中变化，放在函数中设置
 * 注：窗口要包含中心点，故 M 和 N 为奇数 
 */
constexpr double Threshold = 0.5;
constexpr double Threshold_Crack = 0.1;
constexpr double Threshold_Background = 0.8;
constexpr double Diff_Background = 0;
constexpr int    RM = 20;
constexpr int    M = 2 * RM + 1;

/* 去噪算法的参数
 * Threshold_FcMax           渗流阈值，Fc < 255 * FcMax 则可以渗透
 * Threshold_FcMax_Crack     裂缝环形度 Fc 上限，Fc >  255 * T 则认为是裂缝
 * Threshold_RadiusMin       裂缝半径长 R  下限，R  <  T       则认为是裂缝
 * 注：上述两个判别都不成立时认为是背景
 * 注：判别点的渗流区域（Dp）标记为相同成分
 */
constexpr double Threshold_FcMax = 0.4;
constexpr double Threshold_FcMax2 = 0.4;
constexpr double Threshold_RadiusMax = 10;


//// 模块内部辅助变量
/* localWindow
 *  0x00  ==>  不在 Dp 中，也不在 Dc 中
 *  0xf0  ==>  只在 Dp 中
 *  0x0f  ==>  只在 Dc 中，还未加入 Dp
 *  0xff  ==>  曾在 Dc 中, 现在 Dp 中
 *  注：只有第一种情况下可能将点加入 Dc
 */
enum localWindow : unsigned char {
    NOT_PICK      = 0x00,
    ONLY_IN_DP    = 0xf0,
    ONLY_IN_DC    = 0x0f,
    FROM_DC_TO_DP = 0xff,
};
// 用于设置 localWindow 的值
constexpr uchar IN_DP = 0xf0;
constexpr uchar IN_DC = 0x0f;

/* marks
 * 0x00  ==>  未标记  ==>  设为 255 * Fc  （？） ==> --- 
 * 0x0f  ==>  背景点  ==>  设为 255       （白） ==> 255
 * 0xf0  ==>  裂缝点  ==>  设为 255 * Fc  （暗） ==> 0   
 * 0xff  ==>  混淆点  ==>  设为 127       （中） ==> 255 
 * 注：在至少两次渗流中被标记为不同成分的点认为其为混淆点
 * 注：第一列设置用于 percolation，第二列用于 denoise
 */
enum marks : unsigned char {
    NOT_MARK      = 0x00,
    BACKGROUND    = 0xf0, 
    CRACK         = 0x0f,
    CONFUSED      = 0xff,
};
// 用于设置 localWindow 的值
constexpr uchar IS_BACKGROUND = 0xf0;
constexpr uchar IS_CRACK      = 0x0f;


//// 模块内部辅助函数
// 涉及到名称固定的、不方便作为函数参数的外部变量的宏“函数”
#define setLocalWindow(point, val)  localWindow[point.x - luPoint.x][point.y - luPoint.y] |= val
#define getLocalWindow(point)       localWindow[point.x - luPoint.x][point.y - luPoint.y]
#define setMarks(point, val)        marks[point.x][point.y] |= val
#define getMarks(point)             marks[point.x][point.y] |= val
#define getGrayVal(point)           img.at<uchar>(point.x, point.y)

// 可能会做修改的、多处使用的函数
inline auto calculateFc(const auto& Dp) {
    return (Dp.size()) / (M * 1.0 * M);
}

// 用于调试的函数
inline auto showRegion(const cv::Mat& img, const auto& Dp, const auto& center, double Fc = 0, int time=0) {
    auto midImg = img.clone();
    for (const auto& p : Dp) {
        midImg.at<uchar>(p.corr.x, p.corr.y) = 0;
    }
    cv::putText(midImg, "Fc=" + to_string(Fc), { 50, 50 }, 5, 1, 1);
    cv::imshow("mid result", midImg);
    cv::waitKey(time);
    return Fc;
}

inline auto showRegionAndRadius(const cv::Mat& img, const auto& Dp, const auto& center, double Fc = 0, double Radius = 0, int time = 0) {
    auto midImg = img.clone();
    for (const auto& p : Dp) {
        midImg.at<uchar>(p.corr.x, p.corr.y) = 0;
    }
    cv::putText(midImg, "Fc=" + to_string(Fc) + " R=" + to_string(Radius), { 50, 50 }, 5, 1, 1);
    cv::imshow("mid result", midImg);
    cv::waitKey(time);
    return Fc;
}

// 书中描述的算法
// 注：加速算法会在计算中修改图像灰度值，令其修改 img_out 以保持 img 的信息
auto percolation_single_pixel(const cv::Mat& img, cv::Mat& img_out, auto& marks, int r, int c) -> decltype(auto) {
    // 中心像素坐标
    Corr center(r, c);
    
    // 初始化参数
    auto T = getGrayVal(center);
    auto w = 1;
    auto RN = RM / 2;
    auto N = 2 * RN + 1;
    auto isOutofWindow = false;

    // 初始化 Dp 和 Dc
    // 一个 trick，设置 Dp 依灰度值由大到小，而 Dc 由小到大
    // 1. 更新 T  时直接取 Dp.begin() 比较
    // 2. 更新 Dp 时做找到第一个大于 T 的点即可（变体二分查找）
    CorrWithVal<uchar> center_point(T, center);
    using point_type = decltype(center_point);
    set<point_type, greater<point_type>> Dp{};
    set<point_type> Dc{};
    Dp.insert(center_point);

    array< array<uchar, M>, M> localWindow{};
    // localWindow 左上角坐标
    auto luPoint = center - Corr{RM, RM};

    while (!isOutofWindow) {
        T = max(Dp.begin()->val, T) + w;
        // 遍历 Dp，对每个像素点遍历其八邻域，更新 Dc
        for (const auto& p : Dp) {
            for (const auto& direction : directionTable) {
                auto neiborPoint = p.corr + direction;
                if (getLocalWindow(neiborPoint) == localWindow::NOT_PICK) {
                    auto val = getGrayVal(neiborPoint);
                    Dc.insert({ val, neiborPoint });
                    setLocalWindow(neiborPoint, IN_DC);
                }
            }
        }

        // 渗流阶段 1.
        // 若没有可渗流的像素点，即： upIter == Dc.begin()，则选择 Dc.begin() 做为渗流点
        auto upIter = Dc.upper_bound({ T, {-1, -1} });
        if (upIter == Dc.begin()) 
            upIter++;
        Dp.insert(Dc.begin(), upIter);
        // 设置新加入的点的状态，同时检查是否超出区域
        for (auto lt = Dc.begin(); lt != upIter; lt++) {
            setLocalWindow(lt->corr, IN_DP);
            auto disp = lt->corr - center;
            if (abs(disp.x) > RN || abs(disp.y) > RN) {
                isOutofWindow = true;
            }
        }
        Dc.erase(Dc.begin(), upIter);
    } // while

    // 超出最小窗口时，进入下一阶段并更新 N
    RN++; N += 2;
    isOutofWindow = false;
    while (N < M) {
        T = max(Dp.begin()->val, T) + w;

        // 遍历 Dp，对每个像素点，遍历其八邻域，更新 Dc
        for (const auto& p : Dp) {
            for (const auto& direction : directionTable) {
                auto neiborPoint = p.corr + direction;
                if (getLocalWindow(neiborPoint) == localWindow::NOT_PICK) {
                    auto val = getGrayVal(neiborPoint);
                    Dc.insert({ val, neiborPoint });
                    setLocalWindow(neiborPoint, IN_DC);
                }
            }
        }

        // 渗流阶段 2.
        // 当无法渗透时，终止渗流
        auto upIter = Dc.upper_bound({ T, {-1, -1} });
        if (upIter == Dc.begin()) {
            auto Fc = calculateFc(Dp);
            //showRegion(img, Dp, center, Fc, 0);
            if (Fc < Threshold_Crack) {
                for (const auto& p : Dp) {
                    setMarks(p.corr, IS_CRACK);
                    img_out.at<uchar>(p.corr.x, p.corr.y) = 255 * Fc;
                }
            }
            else if (Fc >= Threshold_Background) {
                for (const auto& p : Dp) {
                    // 使用 int 避免越界 (255) 
                    int val = center_point.val + Diff_Background;
                    if (p.val >= val) {
                        setMarks(p.corr, IS_BACKGROUND);
                    }
                }
            }
            return Fc;
        }
        Dp.insert(Dc.begin(), upIter);
        // 设置新加入的点的状态，同时检查是否超出区域
        for (auto lt = Dc.begin(); lt != upIter; lt++) {
            setLocalWindow(lt->corr, IN_DP);
            auto vec = lt->corr - center;
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
    } // while

    // 超出最大窗口后，终止渗流
    auto Fc = calculateFc(Dp);
    //showRegion(img, Dp, center, Fc, 0);
    if (Fc < Threshold_Crack) {
        for (const auto& p : Dp) {
            setMarks(p.corr, IS_CRACK);
            img_out.at<uchar>(p.corr.x, p.corr.y) = 255 * Fc;
        }
    }
    else if (Fc > Threshold_Background) {
        for (const auto& p : Dp) {
            int val = center_point.val + Diff_Background;
            if (p.val >= val) {
                setMarks(p.corr, IS_BACKGROUND);
            }
        }
    }
    return Fc;
} // calculateFc

auto denoise_single_pixel(const cv::Mat& img, auto& marks, int r, int c) {
    // 中心像素坐标
    Corr center(r, c);

    // 初始化参数
    auto T = getGrayVal(center);

    // 初始化 Dp 和 Dc
    CorrWithVal<uchar> center_point(T, center);
    using point_type = decltype(center_point);
    set<point_type, greater<point_type>> Dp{};
    set<point_type> Dc{};
    Dp.insert(center_point);

    auto rows = img.rows;
    auto cols = img.cols;

    vector<vector<uchar>> localWindow(rows, vector<uchar>(cols, localWindow::NOT_PICK));
    // localWindow 起点坐标
    auto luPoint = Corr{ 0, 0 };

    while (true) {
        // 遍历 Dp，对每个像素点遍历其八邻域，更新 Dc
        for (const auto& p : Dp) {
            for (const auto& direction : directionTable) {
                auto neiborPoint = p.corr + direction;
                if (neiborPoint.x < 0 || neiborPoint.x >= rows || 
                    neiborPoint.y < 0 || neiborPoint.y >= cols)
                    continue;
                if (getLocalWindow(neiborPoint) == localWindow::NOT_PICK) {
                    auto val = getGrayVal(neiborPoint);
					Dc.insert({ val, neiborPoint });
                    setLocalWindow(neiborPoint, IN_DC);
                }
            }
        }

        // 渗流
        auto upIter = Dc.upper_bound({ static_cast<uchar>(255 * Threshold_FcMax), {-1, -1} });
        if (upIter == Dc.begin()) {
            auto radius = 0;
            for (const auto& p : Dp) {
                auto disp = p.corr - center;
                radius = max(radius, max(abs(disp.x), abs(disp.y)));
            }
            auto Fc = (Dp.size()) / (4.0 * radius * radius);
            //showRegionAndRadius(img, Dp, center, Fc, radius, 0);
            if (Fc > Threshold_FcMax2 || radius < Threshold_RadiusMax) {
                for (const auto& p : Dp) {
                    setMarks(p.corr, IS_BACKGROUND);
                }
            } 
            else {
                for (const auto& p : Dp) {
                    setMarks(p.corr, IS_CRACK);
                }
            }
            return; 
        }
        Dp.insert(Dc.begin(), upIter);
        for (auto lt = Dc.begin(); lt != upIter; lt++) {
            setLocalWindow(lt->corr, IN_DP);
        }
        Dc.erase(Dc.begin(), upIter);
    } // while
} // denoise


export auto percolation(cv::Mat& img) {
    auto rows = img.rows;
    auto cols = img.cols;
    vector<vector<uchar>> marks(rows + M, vector<uchar>(cols + M, marks::NOT_MARK));

    // 填充图像以省去越界检查，镜像填充以保持边框的渗流
    cv::Mat img_temp;
    cv::copyMakeBorder(img, img_temp, RM, RM, RM, RM, cv::BORDER_DEFAULT);
        
    // 减一点亮度避免周围都是 255 白点无法渗流导致无法加速
    cv::Mat img_out = img_temp.clone();
    img_out = img_out - 1;

    for (int r = RM; r < rows + RM; r++) {
        printf("current rows = %d\n", r - RM);
        for (int c = RM; c < cols + RM; c++) {
            if (marks[r][c] == marks::NOT_MARK) {
				auto Fc = percolation_single_pixel(img_temp, img_out, marks, r, c);
				img_out.at<uchar>(r, c) = Fc * 255;
            }
        }
    }
    for (int r = RM; r < rows + RM; r++) {
        for (int c = RM; c < cols + RM; c++) {
            switch (marks[r][c]) {
                case marks::BACKGROUND: { img_out.at<uchar>(r, c) = 255; break; }
                case marks::CRACK:      { /* 其值在函数中被设置 255*TMax */ break; }
                case marks::CONFUSED:   { img_out.at<uchar>(r, c) = 255; break; }
                case marks::NOT_MARK:   { /* 其值在循环中被设置 255*TMax */ break; }
                default:
                    assert(true);
                    exit(-1);
            }
        }
    }
    // 去掉之前的填充
    auto width = cols;
    auto height = rows;
    img = img_out(cv::Rect(RM, RM, width, height)).clone();
}

export auto denoise(cv::Mat& img) {
    auto rows = img.rows;
    auto cols = img.cols;
    vector<vector<uchar>> marks(rows, vector<uchar>(cols, marks::NOT_MARK));

    cv::Mat img_out = img.clone();

    for (int r = 0; r < rows; r++) {
        printf("current rows = %d\n", r);
        for (int c = 0; c < cols; c++) {
            auto val = img_out.at<uchar>(r, c);
            if (val <= 255 * Threshold_FcMax && marks[r][c] == marks::NOT_MARK)
                denoise_single_pixel(img_out, marks, r, c);
        }
    }
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            auto val = img_out.at<uchar>(r, c);
            if (val <= 255 * Threshold_FcMax) {
                switch (marks[r][c]) {
					case marks::BACKGROUND: { img_out.at<uchar>(r, c) = 255; break; }
					case marks::CRACK:      { img_out.at<uchar>(r, c) = 0;   break; }
					case marks::CONFUSED:   { img_out.at<uchar>(r, c) = 255; break; }
					case marks::NOT_MARK:   { img_out.at<uchar>(r, c) = 0;   break; }
					default:
						assert(true);
						exit(-1);
                }
            }
            else {
                img_out.at<uchar>(r, c) = 255;
            }
        }
    }
    img = img_out.clone();
}
