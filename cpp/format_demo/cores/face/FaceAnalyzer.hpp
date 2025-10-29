#ifndef FACEEMBEDDINGEXTRACTOR_FACEANALYZER_HPP
#define FACEEMBEDDINGEXTRACTOR_FACEANALYZER_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

struct Face {
    cv::Rect2d bbox;
    float det_score = 0.0f;
    std::vector<cv::Point2f> kps;
    cv::Mat embedding;
    cv::Mat aligned_face;

    // 添加默认构造函数以方便使用
    Face() = default;
};

class FaceAnalyzer {
public:
    FaceAnalyzer(const std::string &det_model_path, const std::string &rec_model_path);

    void prepare(const std::string &provider, float det_thresh, cv::Size det_size, bool use_fp16 = false);

    std::vector<Face> get(const cv::Mat &img);

    std::vector<Face> detect(const cv::Mat &img);

    // ======================= 【新增功能】 =======================
    // 新增: 从完整图像和Face对象（含kps）提取特征, 填充face.embedding
    void get_embedding(const cv::Mat &full_img, Face &face);
    // ======================= 【修改结束】 =======================

private:
    cv::Mat get_embedding_from_aligned(const cv::Mat &aligned_img);

    cv::Mat norm_crop(const cv::Mat &img, const std::vector<cv::Point2f> &landmark, cv::Size crop_size = {112, 112});

    cv::dnn::Net det_net_;
    cv::dnn::Net rec_net_;
    float det_thresh_ = 0.5f;
    cv::Size det_size_ = {640, 640};
};

#endif //FACEEMBEDDINGEXTRACTOR_FACEANALYZER_HPP