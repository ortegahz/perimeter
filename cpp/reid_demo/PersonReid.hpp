// include/PersonReid.h

#ifndef PERSON_REID_H
#define PERSON_REID_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

class PersonReid {
public:
    /**
     * @brief Constructor for the PersonReid class.
     *
     * @param onnx_model_path Path to the .onnx model file.
     * @param input_width The width of the model's input tensor.
     * @param input_height The height of the model's input tensor.
     * @param use_gpu Flag to enable CUDA backend for inference.
     */
    PersonReid(const std::string &onnx_model_path, int input_width, int input_height, bool use_gpu = true);

    /**
     * @brief Extracts a feature vector from a single image.
     * It mimics the Python script's logic: inference on original + flipped, sum, and normalize.
     *
     * @param image The input image (cv::Mat, assumes BGR color format from cv::imread).
     * @return cv::Mat The normalized feature vector (1xN dimensions). Returns an empty Mat on failure.
     */
    cv::Mat extract_feat(const cv::Mat &image);

private:
    /**
     * @brief Internal function to preprocess an image and run a single inference pass.
     *
     * @param image The input image.
     * @param flip True to flip the image horizontally before processing.
     * @return cv::Mat The raw feature vector from the model.
     */
    cv::Mat run_inference(const cv::Mat &image, bool flip);

private:
    cv::dnn::Net net_;
    cv::Size input_size_;
    // Normalization parameters from PyTorch's ImageNet normalization (for RGB)
    cv::Scalar mean_ = {0.485, 0.456, 0.406};
    cv::Scalar std_ = {0.229, 0.224, 0.225};
};

#endif // PERSON_REID_H