module;

#include <opencv2/opencv.hpp>  
#include <utility>
#include <iostream>

export module percolation;

using namespace std;

import Corr;
import DEBUG;

//// ����
constexpr double PI = 3.14159265;
const Corr directionTable[8] = {
    {-1, -1}, {-1, 0}, {-1, 1},
    { 0, -1}, { 0, 1},
    { 1, -1}, { 1, 0},  {1, 1},
};


//// �㷨���� ==> �������ɽṹ�壬������û��Ҫ
/* �����㷨�Ĳ���
 * Threshold                ԭ�㷨�ж�����ֵ��Fc < T ����Ϊ�ѷ죬������Ϊ����
 * Threshold_Crack          �����㷨�ж��ѷ����ֵ��Fc <  T ����Ϊ��������Dp�������ѷ�
 * Threshold_Background     �����㷨�ж���������ֵ��Fc >= T ����Ϊ��������Dp�������Ǳ�����ȡ���ڻҶȵĲ�ֵ
 * Diff_Background          �����㷨�ж������Ĳ�ֵ��Val - Fc >= D ʱ��Ϊ����
 * N, M, RN, RM             ������С����󣩴��ڴ�С����뾶
 * ע��N �ڼ����б仯�����ں���������
 * ע������Ҫ�������ĵ㣬�� M �� N Ϊ���� 
 */
constexpr double Threshold = 0.5;
constexpr double Threshold_Crack = 0.1;
constexpr double Threshold_Background = 0.8;
constexpr double Diff_Background = 0;
constexpr int    RM = 20;
constexpr int    M = 2 * RM + 1;

/* ȥ���㷨�Ĳ���
 * Threshold_FcMax           ������ֵ��Fc < 255 * FcMax �������͸
 * Threshold_FcMax_Crack     �ѷ컷�ζ� Fc ���ޣ�Fc >  255 * T ����Ϊ���ѷ�
 * Threshold_RadiusMin       �ѷ�뾶�� R  ���ޣ�R  <  T       ����Ϊ���ѷ�
 * ע�����������б𶼲�����ʱ��Ϊ�Ǳ���
 * ע���б�����������Dp�����Ϊ��ͬ�ɷ�
 */
constexpr double Threshold_FcMax = 0.4;
constexpr double Threshold_FcMax2 = 0.4;
constexpr double Threshold_RadiusMax = 10;

/* �ص����ڵĲ���
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


//// ģ���ڲ���������
/* localWindow
 *  0x00  ==>  ���� Dp �У�Ҳ���� Dc ��
 *  0xf0  ==>  ֻ�� Dp ��
 *  0x0f  ==>  ֻ�� Dc �У���δ���� Dp
 *  0xff  ==>  ���� Dc ��, ���� Dp ��
 *  ע��ֻ�е�һ������¿��ܽ������ Dc
 */
enum localWindow : unsigned char {
    NOT_PICK      = 0x00,
    ONLY_IN_DP    = 0xf0,
    ONLY_IN_DC    = 0x0f,
    FROM_DC_TO_DP = 0xff,
};
// �������� localWindow ��ֵ
constexpr uchar IN_DP = 0xf0;
constexpr uchar IN_DC = 0x0f;

/* marks
 * 0x00  ==>  δ���  ==>  ��Ϊ 255 * Fc  ������ ==> --- 
 * 0x0f  ==>  ������  ==>  ��Ϊ 255       ���ף� ==> 255
 * 0xf0  ==>  �ѷ��  ==>  ��Ϊ 255 * Fc  ������ ==> 0   
 * 0xff  ==>  ������  ==>  ��Ϊ 127       ���У� ==> 255 
 * ע�����������������б����Ϊ��ͬ�ɷֵĵ���Ϊ��Ϊ������
 * ע����һ���������� percolation���ڶ������� denoise
 */
enum marks : unsigned char {
    NOT_MARK      = 0x00,
    BACKGROUND    = 0xf0, 
    CRACK         = 0x0f,
    CONFUSED      = 0xff,
};
// �������� localWindow ��ֵ
constexpr uchar IS_BACKGROUND = 0xf0;
constexpr uchar IS_CRACK      = 0x0f;


//// ģ���ڲ���������
// �漰�����ƹ̶��ġ���������Ϊ�����������ⲿ�����ĺꡰ������
// TODO ����д�Ĳ��ã�set ʵ��������λ���������� set 0x00 �Ĳ��������У������˻���
#define setLocalWindow(point, val)  localWindow[point.x - luPoint.x][point.y - luPoint.y] |= val
#define getLocalWindow(point)       localWindow[point.x - luPoint.x][point.y - luPoint.y]
#define setMarks(point, val)        marks[point.x][point.y] |= val
#define getMarks(point)             marks[point.x][point.y]
#define resetMarks(point)           marks[point.x][point.y] == marks::NOT_MARK
#define getGrayVal(point)           img.at<uchar>(point.x, point.y)

// ���ܻ����޸ĵġ��ദʹ�õĺ���
inline auto calculateFc(const auto& Dp) {
    return (Dp.size()) / (M * 1.0 * M);
}

// ���ڵ��Եĺ���
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

// �����������㷨
// ע�������㷨���ڼ������޸�ͼ��Ҷ�ֵ�������޸� img_out �Ա��� img ����Ϣ
auto percolation_single_pixel(const cv::Mat& img, 
                                    cv::Mat& img_out, 
                                    vector<vector<uchar>>& marks, 
                                    int r, 
                                    int c ) {
    // ������������
    Corr center(r, c);
    
    // ��ʼ������
    auto T = getGrayVal(center);
    auto w = 1;
    auto RN = RM / 2;
    auto N = 2 * RN + 1;
    auto isOutofWindow = false;

    // ��ʼ�� Dp �� Dc
    // һ�� trick������ Dp ���Ҷ�ֵ�ɴ�С���� Dc ��С����
    // 1. ���� T  ʱֱ��ȡ Dp.begin() �Ƚ�
    // 2. ���� Dp ʱ���ҵ���һ������ T �ĵ㼴�ɣ�������ֲ��ң�
    CorrWithVal<uchar> center_point(T, center);
    using point_type = decltype(center_point);
    set<point_type, greater<point_type>> Dp{};
    set<point_type> Dc{};
    Dp.insert(center_point);

    array< array<uchar, M>, M> localWindow{};
    // localWindow ���Ͻ�����
    auto luPoint = center - Corr{RM, RM};

    while (!isOutofWindow) {
        T = max(Dp.begin()->val, T) + w;
        // ���� Dp����ÿ�����ص����������򣬸��� Dc
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

        // �����׶� 1.
        // ��û�п����������ص㣬���� upIter == Dc.begin()����ѡ�� Dc.begin() ��Ϊ������
        auto upIter = Dc.upper_bound({ T, {-1, -1} });
        if (upIter == Dc.begin()) 
            upIter++;
        Dp.insert(Dc.begin(), upIter);
        // �����¼���ĵ��״̬��ͬʱ����Ƿ񳬳�����
        for (auto lt = Dc.begin(); lt != upIter; lt++) {
            setLocalWindow(lt->corr, IN_DP);
            auto disp = lt->corr - center;
            if (abs(disp.x) > RN || abs(disp.y) > RN) {
                isOutofWindow = true;
            }
        }
        Dc.erase(Dc.begin(), upIter);
    } // while

    // ������С����ʱ��������һ�׶β����� N
    RN++; N += 2;
    isOutofWindow = false;
    while (N < M) {
        T = max(Dp.begin()->val, T) + w;

        // ���� Dp����ÿ�����ص㣬����������򣬸��� Dc
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

        // �����׶� 2.
        // ���޷���͸ʱ����ֹ����
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
                    // ʹ�� int ����Խ�� (255) 
                    int val = center_point.val + Diff_Background;
                    if (p.val >= val) {
                        setMarks(p.corr, IS_BACKGROUND);
                    }
                }
            }
            return Fc;
        }
        Dp.insert(Dc.begin(), upIter);
        // �����¼���ĵ��״̬��ͬʱ����Ƿ񳬳�����
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

    // ������󴰿ں���ֹ����
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
    // ������������
    Corr center(r, c);

    // ��ʼ������
    auto T = getGrayVal(center);

    // ��ʼ�� Dp �� Dc
    CorrWithVal<uchar> center_point(T, center);
    using point_type = decltype(center_point);
    set<point_type, greater<point_type>> Dp{};
    set<point_type> Dc{};
    Dp.insert(center_point);

    auto rows = img.rows;
    auto cols = img.cols;

    vector<vector<uchar>> localWindow(rows, vector<uchar>(cols, localWindow::NOT_PICK));
    // localWindow �������
    auto luPoint = Corr{ 0, 0 };

    while (true) {
        // ���� Dp����ÿ�����ص����������򣬸��� Dc
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

        // ����
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

    // ���ͼ����ʡȥԽ���飬��������Ա��ֱ߿������
    cv::Mat img_temp;
    cv::copyMakeBorder(img, img_temp, RM, RM, RM, RM, cv::BORDER_DEFAULT);
        
    // ��һ�����ȱ�����Χ���� 255 �׵��޷����������޷�����
    img_temp = img_temp - 5;
    cv::Mat img_out = img_temp.clone();
    img_out = img_out - 5;
  
    for (int r = RM; r < rows + RM; r++) {
        printf("current rows = %d\n", r - RM);
        for (int c = RM; c < cols + RM; c++) {
            if (marks[r][c] == marks::NOT_MARK) {
				auto Fc = percolation_single_pixel(img_temp, img_out, marks, r, c);
				img_out.at<uchar>(r, c) = Fc * 255;
                // ������ֱ���϶��ɷ֣���ع�ԭʼ�㷨��ʹ����ֵ�ж�
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
                case marks::CRACK:      { /* ��ֵ�ں����б����� 255*TMax */ break; }
                case marks::CONFUSED:   { img_out.at<uchar>(r, c) = 255; break; }
                //case marks::NOT_MARK:   { /* ��ֵ��ѭ���б����� 255*TMax */ break; }
                default:
                    assert(true);
                    exit(-1);
            }
        }
    }
    // ȥ��֮ǰ�����
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
    // len Ϊ���ڵ�Ԫ�߳���L Ϊ�ص����ڱ߳�
    auto L = C_ * len_;

    // ����֮��ָ�ͼ���С
    auto [rows, cols] = std::make_tuple(img.rows, img.cols);
    auto [h_res, w_res] = std::make_tuple(rows % len_, cols % len_);
    if (h_res == 0) h_res = len_;
    if (w_res == 0) w_res = len_;
    // Ϊ��С��ƥ��Ĵ��ڵ�Ԫ�߳���ͼ�����
    cv::copyMakeBorder(img, img, 0, len_ - h_res, 0, len_ - w_res, cv::BORDER_DEFAULT);
    // Ϊ��֤ͼ���ڵĴ��ڵ�Ԫȷʵ�� "���" 3 * 3 = 9 �����
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
    // ȥ���������
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
    
	// ͳ���ѷ�� (marks ����ϵ��
	crack_vec.clear();
	for (auto i = 0; i < rows; i++) {
		for (auto j = 0; j < cols; j++) {
			Corr corr(i + RM, j + RM);
			if (getMarks(corr) == marks::CRACK) {
				crack_vec.push_back(corr);
			}
		}
	}
	// ͳ�Ʒ��ѷ���ھӣ����� �ȴ������ĵ㼯 ��
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
        // ����������������
        percolation(img, marks);
        showTemp(img);

        // ����Ƿ����µĵ㱻���Ϊ�ѷ�
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
