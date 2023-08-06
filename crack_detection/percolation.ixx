module;

#include <opencv2/opencv.hpp>  
#include <utility>
#include <iostream>

export module percolation;

using namespace std;
import Corr;

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
#define setLocalWindow(point, val)  localWindow[point.x - luPoint.x][point.y - luPoint.y] |= val
#define getLocalWindow(point)       localWindow[point.x - luPoint.x][point.y - luPoint.y]
#define setMarks(point, val)        marks[point.x][point.y] |= val
#define getMarks(point)             marks[point.x][point.y] |= val
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
auto percolation_single_pixel(const cv::Mat& img, cv::Mat& img_out, auto& marks, int r, int c) -> decltype(auto) {
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

auto denoise_single_pixel(const cv::Mat& img, auto& marks, int r, int c) {
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


export auto percolation(cv::Mat& img) {
    auto rows = img.rows;
    auto cols = img.cols;
    vector<vector<uchar>> marks(rows + M, vector<uchar>(cols + M, marks::NOT_MARK));

    // ���ͼ����ʡȥԽ���飬��������Ա��ֱ߿������
    cv::Mat img_temp;
    cv::copyMakeBorder(img, img_temp, RM, RM, RM, RM, cv::BORDER_DEFAULT);
        
    // ��һ�����ȱ�����Χ���� 255 �׵��޷����������޷�����
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
                case marks::CRACK:      { /* ��ֵ�ں����б����� 255*TMax */ break; }
                case marks::CONFUSED:   { img_out.at<uchar>(r, c) = 255; break; }
                case marks::NOT_MARK:   { /* ��ֵ��ѭ���б����� 255*TMax */ break; }
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
