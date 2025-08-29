#include "PersonReid.hpp"
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

PersonReid::PersonReid(const std::string &onnx_model_path, int input_width, int input_height, bool use_gpu) {
    this->input_size_ = cv::Size(input_width, input_height);

    this->net_ = cv::dnn::readNetFromONNX(onnx_model_path);
    if (this->net_.empty()) {
        std::cerr << "FATAL: Failed to load ONNX model from " << onnx_model_path << std::endl;
        throw std::runtime_error("Failed to load ONNX model");
    }

    if (use_gpu) {
        std::cout << "Attempting to use CUDA backend..." << std::endl;
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

cv::Mat PersonReid::extract_feat(const cv::Mat &image) {
    cv::Mat feat_orig = this->run_inference(image, false);
    cv::Mat feat_flipped = this->run_inference(image, true);

    if (feat_orig.empty() || feat_flipped.empty()) {
        std::cerr << "Warning: Inference returned an empty feature vector." << std::endl;
        return cv::Mat();
    }

    cv::Mat feat_sum = feat_orig + feat_flipped;

    cv::Mat feat_norm;
    cv::normalize(feat_sum, feat_norm, 1.0, 0.0, cv::NORM_L2);

    return feat_norm;
}

cv::Mat PersonReid::run_inference(const cv::Mat &image, bool flip) {
    // 保证 3 通道
    cv::Mat img_3ch;
    if (image.channels() == 1) {
        cv::cvtColor(image, img_3ch, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, img_3ch, cv::COLOR_BGRA2BGR);
    } else {
        img_3ch = image.clone();
    }

    // BGR -> RGB
    cv::Mat img_rgb;
    cv::cvtColor(img_3ch, img_rgb, cv::COLOR_BGR2RGB);

    // 这里不再 resize，假设输入图片已经 Python 端 resize 到目标尺寸
    if (img_rgb.cols != input_size_.width || img_rgb.rows != input_size_.height) {
        std::cerr << "Warning: Input size (" << img_rgb.cols << "x" << img_rgb.rows
                  << ") does not match model input size (" << input_size_.width << "x" << input_size_.height << ")"
                  << std::endl;
    }

    // Optional flip
    if (flip) {
        cv::flip(img_rgb, img_rgb, 1);
    }

    // 转 float32 并归一化到 [0,1]
    cv::Mat img_float;
    img_rgb.convertTo(img_float, CV_32F, 1.0 / 255.0);

    // 手动标准化
    std::vector<cv::Mat> channels(3);
    cv::split(img_float, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f;  // R
    channels[1] = (channels[1] - 0.456f) / 0.224f;  // G
    channels[2] = (channels[2] - 0.406f) / 0.225f;  // B
    cv::Mat normalized;
    cv::merge(channels, normalized);

    // HWC -> NCHW
    cv::Mat blob = cv::dnn::blobFromImage(normalized, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);

    // 推理
    this->net_.setInput(blob);
    cv::Mat output = this->net_.forward();
    return output;
}