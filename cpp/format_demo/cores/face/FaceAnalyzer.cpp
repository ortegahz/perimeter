#include "FaceAnalyzer.hpp"
#include <iostream>
#include <stdexcept>
#include <map>
#include <numeric>
#include <chrono>

// FOR EXPERIMENT: 实现新函数
cv::Mat FaceAnalyzer::get_embedding_from_aligned(const cv::Mat &aligned_img) {
    if (aligned_img.size() != cv::Size(112, 112)) {
        throw std::runtime_error("Input for get_embedding_from_aligned must be a 112x112 image.");
    }
    cv::Mat blob = cv::dnn::blobFromImage(aligned_img, 1.0 / 128.0, cv::Size(112, 112),
                                          cv::Scalar(127.5, 127.5, 127.5), true, false);
    rec_net_.setInput(blob);
    return rec_net_.forward().clone();
}

FaceAnalyzer::FaceAnalyzer(const std::string &det_model_path, const std::string &rec_model_path) {
    det_net_ = cv::dnn::readNetFromONNX(det_model_path);
    rec_net_ = cv::dnn::readNetFromONNX(rec_model_path);
    if (det_net_.empty()) {
        throw std::runtime_error("Failed to load detection model: " + det_model_path);
    }
    if (rec_net_.empty()) {
        throw std::runtime_error("Failed to load recognition model: " + rec_model_path);
    }
}

void FaceAnalyzer::prepare(const std::string &provider, float det_thresh, cv::Size det_size) {
    if (provider == "CUDAExecutionProvider") {
        std::cout << "[INFO] Attempting to use CUDA backend." << std::endl;
        det_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        det_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        rec_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        rec_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        std::cout << "[INFO] Using CPU backend." << std::endl;
        det_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        det_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        rec_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        rec_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    det_thresh_ = det_thresh;
    det_size_ = det_size;
}

// -------------------- 修改的 get() --------------------
std::vector<Face> FaceAnalyzer::get(const cv::Mat &img) {
    using clk = std::chrono::high_resolution_clock;

    // 1. 人脸检测
    auto t0 = clk::now();
    auto detected_faces = detect(img);
    auto t1 = clk::now();
    double t_det = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double t_align = 0.0;
    double t_rec = 0.0;

    // 2. 对齐 + 特征提取
    for (auto &face: detected_faces) {
        // ---- 对齐 ----
        auto ta0 = clk::now();
        cv::Mat aligned_face;
        std::vector<cv::Point2f> dst_pts = {
                {38.2946f, 51.6963f},
                {73.5318f, 51.5014f},
                {56.0252f, 71.7366f},
                {41.5493f, 92.3655f},
                {70.7299f, 92.2041f}
        };
        cv::Mat M = cv::estimateAffinePartial2D(face.kps, dst_pts);
        cv::warpAffine(img, aligned_face, M, cv::Size(112, 112));
        face.aligned_face = aligned_face.clone();
        auto ta1 = clk::now();
        t_align += std::chrono::duration<double, std::milli>(ta1 - ta0).count();

        // ---- 特征提取 ----
        auto tr0 = clk::now();
        face.embedding = get_embedding_from_aligned(aligned_face);
        auto tr1 = clk::now();
        t_rec += std::chrono::duration<double, std::milli>(tr1 - tr0).count();
    }

    auto t2 = clk::now();
    double total = std::chrono::duration<double, std::milli>(t2 - t0).count();

    std::cout << "[FaceAnalyzer] det=" << t_det
              << " ms, align=" << t_align
              << " ms, rec=" << t_rec
              << " ms, total=" << total
              << " ms" << std::endl;

    return detected_faces;
}

// -------------------- detect() 保持逻辑不变 --------------------
std::vector<Face> FaceAnalyzer::detect(const cv::Mat &img) {
    cv::Mat input_blob;
    float scale = 1.0f;
    float im_ratio = (float) img.rows / (float) img.cols;
    float model_ratio = (float) det_size_.height / (float) det_size_.width;
    int new_w, new_h;
    if (im_ratio > model_ratio) {
        new_h = det_size_.height;
        new_w = static_cast<int>(new_h / im_ratio);
    } else {
        new_w = det_size_.width;
        new_h = static_cast<int>(new_w * im_ratio);
    }
    scale = (float) new_h / (float) img.rows;
    if (scale == 0) scale = 1.0f;
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(new_w, new_h));
    cv::Mat det_img = cv::Mat::zeros(det_size_, img.type());
    resized_img.copyTo(det_img(cv::Rect(0, 0, new_w, new_h)));
    cv::dnn::blobFromImage(det_img, input_blob, 1.0 / 128.0, det_size_, cv::Scalar(127.5, 127.5, 127.5), true, false);

    std::vector<std::string> out_names = {
            "448", "471", "494",
            "451", "474", "497",
            "454", "477", "500"
    };
    std::vector<cv::Mat> outs;
    det_net_.setInput(input_blob);
    det_net_.forward(outs, out_names);

    std::vector<cv::Rect2d> bboxes;
    std::vector<float> scores;
    std::vector<std::vector<cv::Point2f>> all_kps;
    std::vector<int> strides = {8, 16, 32};
    const int fmc = 3;

    for (size_t i = 0; i < strides.size(); ++i) {
        const int stride = strides[i];
        cv::Mat score_feat = outs[i];
        cv::Mat bbox_feat = outs[i + fmc];
        cv::Mat kps_feat = outs[i + 2 * fmc];

        const int height = det_size_.height / stride;
        const int width = det_size_.width / stride;

        const float *score_data = score_feat.ptr<float>();
        const float *bbox_data = bbox_feat.ptr<float>();
        const float *kps_data = kps_feat.ptr<float>();

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int anchor = 0; anchor < 2; ++anchor) {
                    int idx = (y * width + x) * 2 + anchor;
                    float score = score_data[idx];
                    if (score < det_thresh_) continue;

                    const float *bbox_ptr = &bbox_data[idx * 4];
                    const float *kps_ptr = &kps_data[idx * 10];

                    float cx = (float) x * stride;
                    float cy = (float) y * stride;

                    scores.push_back(score);

                    float x1 = (cx - bbox_ptr[0] * stride) / scale;
                    float y1 = (cy - bbox_ptr[1] * stride) / scale;
                    float x2 = (cx + bbox_ptr[2] * stride) / scale;
                    float y2 = (cy + bbox_ptr[3] * stride) / scale;
                    bboxes.emplace_back(x1, y1, x2 - x1, y2 - y1);

                    std::vector<cv::Point2f> kps;
                    for (int k = 0; k < 5; ++k) {
                        float kx = (cx + kps_ptr[k * 2] * stride) / scale;
                        float ky = (cy + kps_ptr[k * 2 + 1] * stride) / scale;
                        kps.emplace_back(kx, ky);
                    }
                    all_kps.push_back(kps);
                }
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, det_thresh_, 0.4, indices);

    std::vector<Face> results;
    for (int idx: indices) {
        Face f;
        f.bbox = bboxes[idx];
        f.det_score = scores[idx];
        f.kps = all_kps[idx];
        results.push_back(f);
    }
    return results;
}