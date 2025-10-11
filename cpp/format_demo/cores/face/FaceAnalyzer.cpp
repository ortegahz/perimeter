#include "FaceAnalyzer.hpp"
#include <iostream>
#include <stdexcept>
#include <map>
#include <numeric>
#include <chrono>

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
    if (provider == "CUDAExecutionProvider" || provider == "GPU") {
        std::cout << "[INFO] Attempting to use CUDA backend for FaceAnalyzer." << std::endl;
        det_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        det_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        rec_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        rec_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        std::cout << "[INFO] Using CPU backend for FaceAnalyzer." << std::endl;
        det_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        det_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        rec_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        rec_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    det_thresh_ = det_thresh;
    det_size_ = det_size;
}

cv::Mat FaceAnalyzer::get_embedding_from_aligned(const cv::Mat &aligned_img) {
    if (aligned_img.size() != cv::Size(112, 112)) {
        throw std::runtime_error("Input for get_embedding_from_aligned must be a 112x112 image.");
    }

    // --- Blob 预处理计时开始 ---
    auto blob_start = std::chrono::high_resolution_clock::now();
    cv::Mat blob = cv::dnn::blobFromImage(aligned_img, 1.0 / 128.0, cv::Size(112, 112), cv::Scalar(127.5, 127.5, 127.5),
                                          true, false);
    auto blob_end = std::chrono::high_resolution_clock::now();

    // --- setInput 计时开始 ---
    auto set_input_start = std::chrono::high_resolution_clock::now();
    rec_net_.setInput(blob);
    auto set_input_end = std::chrono::high_resolution_clock::now();

    // --- forward 计时开始 ---
    cv::Mat result = rec_net_.forward().clone();
    auto forward_end = std::chrono::high_resolution_clock::now();

    // --- 计算并打印耗时 ---
    auto duration_blob_us = std::chrono::duration_cast<std::chrono::microseconds>(blob_end - blob_start).count();
    auto duration_set_input_us = std::chrono::duration_cast<std::chrono::microseconds>(
            set_input_end - set_input_start).count();
    auto duration_forward_us = std::chrono::duration_cast<std::chrono::microseconds>(
            forward_end - set_input_end).count();

    // 使用更多缩进以表示这是子步骤的耗时
    std::cout << "        [PERF ...from_aligned] Blob: " << duration_blob_us / 1000.0 << "ms, "
              << "SetInput: " << duration_set_input_us / 1000.0 << "ms, "
              << "Forward: " << duration_forward_us / 1000.0 << "ms\n";

    return result;
}

void FaceAnalyzer::get_embedding(const cv::Mat &full_img, Face &face) {
    if (face.kps.size() != 5) {
        throw std::runtime_error("Face object must have 5 keypoints for alignment.");
    }

    // --- 对齐计时开始 ---
    auto align_start = std::chrono::high_resolution_clock::now();

    // 对齐
    const std::vector<cv::Point2f> dst_pts = {
            {38.2946f, 51.6963f},
            {73.5318f, 51.5014f},
            {56.0252f, 71.7366f},
            {41.5493f, 92.3655f},
            {70.7299f, 92.2041f}
    };
    cv::Mat M = cv::estimateAffinePartial2D(face.kps, dst_pts);
    if (M.empty()) {
        face.embedding = cv::Mat(); // 清空 embedding 表示失败
        return;
    }

    cv::Mat aligned_face;
    cv::warpAffine(full_img, aligned_face, M, cv::Size(112, 112));
    face.aligned_face = aligned_face.clone();

    // --- 对齐计时结束 ---
    auto align_end = std::chrono::high_resolution_clock::now();

//    // 保存对齐后的人脸图像
//    static int aligned_face_counter = 0;
//    std::string save_path = "/mnt/nfs/aligned_cpp_bmp/" + std::to_string(aligned_face_counter++) + ".bmp";
//    cv::imwrite(save_path, aligned_face);

    // --- 特征提取计时开始 ---
    auto infer_start = std::chrono::high_resolution_clock::now();

    // 从对齐后的人脸提取特征
    face.embedding = get_embedding_from_aligned(aligned_face);

    // --- 特征提取计时结束 ---
    auto infer_end = std::chrono::high_resolution_clock::now();

    // --- 计算并打印耗时 ---
    auto duration_align_us = std::chrono::duration_cast<std::chrono::microseconds>(align_end - align_start).count();
    auto duration_infer_us = std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start).count();
    auto total_us = duration_align_us + duration_infer_us;

    if (total_us > 1000) { // 仅当总耗时超过1ms时打印，避免刷屏
        std::cout << "      [PERF get_embedding] Total: " << total_us / 1000.0 << "ms | "
                  << "Align: " << duration_align_us / 1000.0 << "ms, "
                  << "Infer(total): " << duration_infer_us / 1000.0 << "ms\n";
    }
}

// -------------------- get() 现在使用新的 get_embedding() 并增加计时 --------------------
std::vector<Face> FaceAnalyzer::get(const cv::Mat &img) {
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    auto detected_faces = detect(img);

    auto t1 = clk::now();
    double t_det = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double t_rec_total = 0.0;

    for (auto &face: detected_faces) {
        try {
            auto tr0 = clk::now();
            get_embedding(img, face); // 使用新方法
            auto tr1 = clk::now();
            t_rec_total += std::chrono::duration<double, std::milli>(tr1 - tr0).count();
        } catch (const std::exception &e) {
            // silent fail
        }
    }

    auto t2 = clk::now();
    double total_time = std::chrono::duration<double, std::milli>(t2 - t0).count();

    // 打印详细耗时
    std::cout << "[FaceAnalyzer::get] Total " << total_time << "ms (det: " << t_det << "ms, rec_all: " << t_rec_total
              << "ms) for " << detected_faces.size() << " faces." << std::endl;

    return detected_faces;
}

// -------------------- detect() 保持逻辑不变 --------------------
std::vector<Face> FaceAnalyzer::detect(const cv::Mat &img) {
    cv::Mat input_blob;
    float scale = 1.0f;
    if (img.empty()) return {};
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

    std::vector<std::string> out_names = {"448", "471", "494", "451", "474", "497", "454", "477", "500"};
    std::vector<cv::Mat> outs;
    det_net_.setInput(input_blob);
    det_net_.forward(outs, out_names);

    std::vector<cv::Rect2d> bboxes;
    std::vector<float> scores;
    std::vector<std::vector<cv::Point2f>> all_kps;
    std::vector<int> strides = {8, 16, 32};

    for (size_t i = 0; i < strides.size(); ++i) {
        const int stride = strides[i];
        const int height = det_size_.height / stride;
        const int width = det_size_.width / stride;

        const float *score_data = outs[i].ptr<float>();
        const float *bbox_data = outs[i + strides.size()].ptr<float>();
        const float *kps_data = outs[i + 2 * strides.size()].ptr<float>();

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int anchor = 0; anchor < 2; ++anchor) {
                    int idx = (y * width + x) * 2 + anchor;
                    float score = score_data[idx];
                    if (score < det_thresh_) continue;

                    // ======================= 【修改的部分在此】 =======================
                    // 将解码逻辑还原为与原始版本一致
                    const float *bbox_ptr = &bbox_data[idx * 4];
                    float cx = (float) x * stride;
                    float cy = (float) y * stride;

                    float x1 = (cx - bbox_ptr[0] * stride) / scale;
                    float y1 = (cy - bbox_ptr[1] * stride) / scale;
                    float x2 = (cx + bbox_ptr[2] * stride) / scale;
                    float y2 = (cy + bbox_ptr[3] * stride) / scale;

                    bboxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
                    scores.push_back(score);

                    std::vector<cv::Point2f> kps;
                    const float *kps_ptr = &kps_data[idx * 10];
                    for (int k = 0; k < 5; ++k) {
                        float kx = (cx + kps_ptr[k * 2] * stride) / scale;
                        float ky = (cy + kps_ptr[k * 2 + 1] * stride) / scale;
                        kps.emplace_back(kx, ky);
                    }
                    all_kps.push_back(kps);
                    // ======================= 【修改结束】 =======================
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