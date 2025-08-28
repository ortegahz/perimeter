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

    // 原始函数，保留以便恢复
    std::vector<Face> get(const cv::Mat &img);

    // ======================= 【修改的部分在此】 =======================
    // FOR EXPERIMENT: 新增一个函数，只对已对齐的112x112图像提取特征
    cv::Mat get_embedding_from_aligned(const cv::Mat &aligned_img);
    // ======================= 【修改结束】 =======================

private:
    std::vector<Face> detect(const cv::Mat &img);

    // (这个函数在此次实验的main中不被调用，但保留)
    cv::Mat norm_crop(const cv::Mat &img, const std::vector<cv::Point2f> &landmark, cv::Size crop_size = {112, 112});

    cv::dnn::Net det_net_;
    cv::dnn::Net rec_net_;
    float det_thresh_ = 0.5f;
    cv::Size det_size_ = {640, 640};
};

#endif //FACEEMBEDDINGEXTRACTOR_FACEANALYZER_HPP