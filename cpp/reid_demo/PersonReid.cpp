#include "PersonReid.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp> // For cv::resize, cv::flip
#include <opencv2/dnn/dnn.hpp> // For cv::dnn::imagesFromBlob

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
        std::cout << "Using CPU backend." << std::endl;
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

cv::Mat PersonReid::extract_feat(const cv::Mat &image) {
    // --- 1. Inference on the original image ---
    cv::Mat feat_orig = this->run_inference(image, false);

    // --- 2. Inference on the flipped image ---
    cv::Mat feat_flipped = this->run_inference(image, true);

    if (feat_orig.empty() || feat_flipped.empty()) {
        std::cerr << "Warning: Inference returned an empty feature vector." << std::endl;
        return cv::Mat();
    }

    // --- 3. Post-processing: sum and normalize ---
    cv::Mat feat_sum = feat_orig + feat_flipped;

    cv::Mat feat_norm;
    cv::normalize(feat_sum, feat_norm, 1.0, 0.0, cv::NORM_L2);

    return feat_norm;
}

cv::Mat PersonReid::run_inference(const cv::Mat &image, bool flip) {
    // --- Preprocessing ---
    cv::Mat input_img;
    if (image.channels() == 1) {
        cv::cvtColor(image, input_img, cv::COLOR_GRAY2BGR);
    } else {
        input_img = image;
    }

    cv::Mat preprocessed_img;
    cv::resize(input_img, preprocessed_img, this->input_size_);

    if (flip) {
        cv::flip(preprocessed_img, preprocessed_img, 1);
    }

    // 3. Create blob with normalization
    cv::Mat blob = cv::dnn::blobFromImage(preprocessed_img, 1.0 / 255.0,
                                          this->input_size_, cv::Scalar(),
                                          true, false, CV_32F);

    // ⚠️ 修正：将 4D NCHW blob 转回 2D+3通道图像后再 split
    std::vector<cv::Mat> imgs;
    cv::dnn::imagesFromBlob(blob, imgs); // NCHW -> vector of HxW 3ch Mats

    if (imgs.empty()) {
        std::cerr << "Error: imagesFromBlob returned empty result." << std::endl;
        return cv::Mat();
    }

    std::vector<cv::Mat> channels;
    cv::split(imgs[0], channels); // 3个通道的Mat

    channels[0] = (channels[0] - this->mean_[0]) / this->std_[0]; // R
    channels[1] = (channels[1] - this->mean_[1]) / this->std_[1]; // G
    channels[2] = (channels[2] - this->mean_[2]) / this->std_[2]; // B

    cv::merge(channels, imgs[0]);

    // 再转回 blob
    blob = cv::dnn::blobFromImage(imgs[0]);

    // --- Inference ---
    this->net_.setInput(blob);
    cv::Mat output = this->net_.forward();

    if (output.empty()) {
        std::cerr << "Warning: net.forward() returned an empty Mat. "
                  << "(Flipped: " << (flip ? "true" : "false") << ")" << std::endl;
    }

    return output;
}