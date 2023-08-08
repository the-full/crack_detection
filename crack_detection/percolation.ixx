module;

#include <opencv2/opencv.hpp>  
#include <utility>
#include <iostream>

export module percolation;

using namespace std;

import Corr;
import DEBUG;

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

/* 重叠窗口的参数
*/
constexpr int C_ = 3;
constexpr int len_ = 50;
constexpr double T_ = 0.05;
constexpr double WT_ = 4.5;
constexpr int T1_ = 8;
constexpr int T2_ = 10;
constexpr int T3_ = 20;
constexpr double P1_ = 0.1;
constexpr double P3_ = 0.1;


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
// TODO 这里写的不好，set 实际上是置位，这让我想 set 0x00 的操作不可行，带来了混乱
#define setLocalWindow(point, val)  localWindow[point.x - luPoint.x][point.y - luPoint.y] |= val
#define getLocalWindow(point)       localWindow[point.x - luPoint.x][point.y - luPoint.y]
#define setMarks(point, val)        marks[point.x][point.y] |= val
#define getMarks(point)             marks[point.x][point.y]
#define resetMarks(point)           marks[point.x][point.y] == marks::NOT_MARK
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
auto percolation_single_pixel(const cv::Mat& img, 
                                    cv::Mat& img_out, 
                                    vector<vector<uchar>>& marks, 
                                    int r, 
                                    int c ) {
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

auto denoise_single_pixel(const cv::Mat& img, 
                                vector<vector<uchar>>& marks, 
                                int r, 
                                int c) {
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


auto percolation(cv::Mat& img, vector<vector<uchar>>& marks) {
    auto rows = img.rows;
    auto cols = img.cols;

    // 填充图像以省去越界检查，镜像填充以保持边框的渗流
    cv::Mat img_temp;
    cv::copyMakeBorder(img, img_temp, RM, RM, RM, RM, cv::BORDER_DEFAULT);
        
    // 减一点亮度避免周围都是 255 白点无法渗流导致无法加速
    img_temp = img_temp - 5;
    cv::Mat img_out = img_temp.clone();
    img_out = img_out - 5;
  
    for (int r = RM; r < rows + RM; r++) {
        printf("current rows = %d\n", r - RM);
        for (int c = RM; c < cols + RM; c++) {
            if (marks[r][c] == marks::NOT_MARK) {
				auto Fc = percolation_single_pixel(img_temp, img_out, marks, r, c);
				img_out.at<uchar>(r, c) = Fc * 255;
                // 若不能直接认定成分，则回归原始算法，使用阈值判定
                if (marks[r][c] == marks::NOT_MARK) {
                    marks[r][c] = Fc < Threshold ? marks::CRACK : marks::BACKGROUND;
                }
            }
        }
    }
    for (int r = RM; r < rows + RM; r++) {
        for (int c = RM; c < cols + RM; c++) {
            assert(marks[r][c] != marks::NOT_MARK);
            switch (marks[r][c]) {
                case marks::BACKGROUND: { img_out.at<uchar>(r, c) = 255; break; }
                case marks::CRACK:      { /* 其值在函数中被设置 255*TMax */ break; }
                case marks::CONFUSED:   { img_out.at<uchar>(r, c) = 255; break; }
                //case marks::NOT_MARK:   { /* 其值在循环中被设置 255*TMax */ break; }
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

auto denoise(cv::Mat& img, vector<vector<uchar>>& marks) {
    auto rows = img.rows;
    auto cols = img.cols;

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

export auto percolation(cv::Mat& img) {
    auto rows = img.rows;
    auto cols = img.cols;
    vector<vector<uchar>> marks(rows + RM * 2, vector<uchar>(cols + RM * 2, marks::NOT_MARK));
    for (auto i = 0; i < rows + RM * 2; i++) {
        for (auto j = 0; j < cols + RM * 2; j++) {
            if (i < RM || i >= rows + RM || j < RM || j >= cols + RM) {
                auto filledPoint = Corr(i, j);
                setMarks(filledPoint, marks::BACKGROUND);
            }
        }
    }
    percolation(img, marks);
}

export auto denoise(cv::Mat& img) {
    auto rows = img.rows;
    auto cols = img.cols;
    vector<vector<uchar>> marks(rows, vector<uchar>(cols, marks::NOT_MARK));
    denoise(img, marks);
}

export auto percolation_accelerate(cv::Mat& img) {
    percolation(img);
    denoise(img);
}

export auto percolation_overlap(cv::Mat& img) {
    // len 为窗口单元边长，L 为重叠窗口边长
    auto L = C_ * len_;

    // 用于之后恢复图像大小
    auto [rows, cols] = std::make_tuple(img.rows, img.cols);
    auto [h_res, w_res] = std::make_tuple(rows % len_, cols % len_);
    if (h_res == 0) h_res = len_;
    if (w_res == 0) w_res = len_;
    // 为大小不匹配的窗口单元边长的图像填充
    cv::copyMakeBorder(img, img, 0, len_ - h_res, 0, len_ - w_res, cv::BORDER_DEFAULT);
    // 为保证图像内的窗口单元确实被 "卷积" 3 * 3 = 9 次填充
    cv::copyMakeBorder(img, img, 2 * len_, 2 * len_, 2 * len_, 2 * len_, cv::BORDER_DEFAULT);
    auto [height, width] = std::make_tuple(img.rows, img.cols);
    auto [h_cnt, w_cnt] = std::make_tuple(height / len_, width / len_);

    cv::Mat weightMatrix(height, width, CV_32FC1, cv::Scalar(0));
    vector<CorrWithVal<float>> temp;
    for (auto h = 0; h < h_cnt - C_ + 1; h++) {
        for (auto w = 0; w < w_cnt - C_ + 1; w++) {
            Corr luPoint{ h * len_, w * len_ };
            auto img_window = img(cv::Rect(luPoint.y, luPoint.x, L, L));
            auto weight_window = weightMatrix(cv::Rect(luPoint.y, luPoint.x, L, L));

            vector<CorrWithVal<uchar>> Points;
            for (auto i = 0; i < L; i++) {
                for (auto j = 0; j < L; j++) {
                    auto val = img_window.at<uchar>(i, j);
                    Points.push_back({ val, i + luPoint.x, j + luPoint.y});
                }
            }
            std::sort(Points.begin(), Points.end());

            int Lp1_num = Points.size() * (1 - 0.3);
            int Lp2_num = Lp1_num * 0.3;
            decltype(Points) Lp1, Lp2;
            Lp1.assign(Points.begin(), Points.begin() + Lp1_num);
            Lp2.assign(Lp1.begin(), Lp1.begin() + Lp2_num);

            double temp_sum = 0.0;
            for (const auto& p : Lp1) {
                temp_sum += p.val;
            }
            double Lmax = std::max_element(Lp1.begin(), Lp1.end())->val;
            double Lmin = std::min_element(Lp2.begin(), Lp2.end())->val;
            double Lavg = temp_sum / Lp1.size();
           
            if (Lavg - Lmin < T1_) {
				for (auto i = 0; i < L; i++) {
					for (auto j = 0; j < L; j++) {
                        temp.push_back({ -0.5, i, j });
					}
				}
            }
            else if (Lavg - Lmin < T2_) {
				for (auto i = 0; i < L; i++) {
					for (auto j = 0; j < L; j++) {
                        temp.push_back({ 0.0, i, j });
					}
				}
            }
            else {
                double P2_;
                if (Lavg - Lmin < T3_) {
		            P2_ = 0.5 + (Lavg - Lmin - T2_) / T3_;
                }
                else {
		            P2_ = 1.0;
                }
                int Num = Points.size();
                int selectNum = Num * P1_ * P2_;
                Points.resize(selectNum);
                auto Dn(std::move(Points));

                for (int rank = 0; rank < selectNum; rank++) {
                    auto corr = Dn[rank].corr;
                    auto W = 1 - static_cast<float>(rank) / Num;
                    temp.push_back({ W, corr });
                }
                for (const auto& p : temp) {
                    weightMatrix.at<float>(p.corr.x, p.corr.y) += p.val;
                }
                temp.clear();
            }
        } // for w
    } // for h
    vector<CorrWithVal<float>> weight_vec;
    // 去掉填充区域
    weightMatrix = weightMatrix(cv::Rect(2 * len_, 2 * len_, cols, rows)).clone();
    img = img(cv::Rect(2 * len_, 2 * len_, cols, rows)).clone();
    for (auto i = 0; i < rows; i++) {
        for (auto j = 0; j < cols; j++) {
            auto w = weightMatrix.at<float>(i, j);
            weight_vec.push_back({ w, i, j });
        }
    }
    using elem_type = decltype(weight_vec.front());
    std::sort(weight_vec.rbegin(), weight_vec.rend());
    int Num = weight_vec.size();
    int extraNum = Num * T_;
    weight_vec.resize(extraNum);
    auto darkPoints(std::move(weight_vec));

    cv::Mat darkImg(rows, cols, CV_8UC1, cv::Scalar(255));
    for (const auto& p : darkPoints) {
        if (p.val < WT_) continue;
        darkImg.at<uchar>(p.corr.x, p.corr.y) = 0;
    }
    denoise(darkImg);

    vector<vector<uchar>> marks(rows + RM * 2, vector<uchar>(cols + RM * 2, marks::BACKGROUND));
    for (auto i = 0; i < rows; i++) {
        for (auto j = 0; j < cols; j++) {
            Corr corr(i + RM, j + RM);
            if (darkImg.at<uchar>(i, j) == 0) {
                marks[corr.x][corr.y] = marks::NOT_MARK;
            }
        }
    }

    percolation(img, marks);

    vector<Corr> crack_vec, wait_percolation_vec;
    vector<Corr> new_crack_vec{};
    
	// 统计裂缝点 (marks 坐标系）
	crack_vec.clear();
	for (auto i = 0; i < rows; i++) {
		for (auto j = 0; j < cols; j++) {
			Corr corr(i + RM, j + RM);
			if (getMarks(corr) == marks::CRACK) {
				crack_vec.push_back(corr);
			}
		}
	}
	// 统计非裂缝点邻居，加入 等待渗流的点集 中
	for (const auto& p : crack_vec) {
		for (const auto& direction : directionTable) {
			auto neiborPoint = p + direction;
			if (getMarks(neiborPoint) != marks::CRACK) {
                resetMarks(neiborPoint);
				wait_percolation_vec.push_back(neiborPoint);
			}
		}
	}

    while (true) {
        // 在新增点上做渗流
        percolation(img, marks);
        showTemp(img);

        // 检查是否有新的点被标记为裂缝
        vector<Corr> temp;
        for (const auto& p : wait_percolation_vec) {
            if (getMarks(p) == marks::CRACK) {
                for (const auto& direction : directionTable) {
                    auto neiborPoint = p + direction;
                    if (getMarks(neiborPoint) != marks::CRACK) {
                        resetMarks(neiborPoint);
                        temp.push_back(neiborPoint);
                    }
                }
            }
        }
        if (temp.size() == 0) 
            break;
        wait_percolation_vec = std::move(temp);
    }

    denoise(img);
}
