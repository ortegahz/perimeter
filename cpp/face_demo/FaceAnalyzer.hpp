#ifndef FACEEMBEDDINGEXTRACTOR_FACEANALYZER_HPP
#define FACEEMBEDDINGEXTRACTOR_FACEANALYZER_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

struct Face {
    cv::Rect2d bbox;
    float det_score;
    std::vector<cv::Point2f> kps;
    cv::Mat embedding;
    cv::Mat aligned_face;
};

class FaceAnalyzer {
public:
    FaceAnalyzer(const std::string &det_model_path, const std::string &rec_model_path);

    void prepare(const std::string &provider, float det_thresh, cv::Size det_size);

    std::vector<Face> get(const cv::Mat &img);

private:
    std::vector<Face> detect(const cv::Mat &img);

    // ======================= 【修改的部分在此】 =======================
    // 添加一个私有辅助函数，用于精确复现insightface的对齐算法
    cv::Mat norm_crop(const cv::Mat &img, const std::vector<cv::Point2f> &landmark);
    // ======================= 【修改结束】 =======================

    cv::dnn::Net det_net_;
    cv::dnn::Net rec_net_;
    float det_thresh_ = 0.5f;
    cv::Size det_size_ = {640, 640};
};

#endif //FACEEMBEDDINGEXTRACTOR_FACEANALYZER_HPP