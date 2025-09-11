#pragma once

#include <opencv2/core/cuda.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <string>

// TensorRT 日志记录器
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override;
};

class PersonReidDLA {
public:
    PersonReidDLA(const std::string &onnx_path,
                  int input_w,
                  int input_h,
                  int dla_core = 0,
                  const std::string &engine_path = "");

    ~PersonReidDLA();

    /**
     * @brief 提取特征向量，直接使用 GpuMat，避免 CPU 拷贝。
     * @param bgr GPU 上的 BGR 图像。
     * @return 1xN 的 GpuMat (CV_32F)，已 L2 归一化。
     */
    cv::cuda::GpuMat extract_feat(const cv::cuda::GpuMat &bgr);

private:
    cv::cuda::GpuMat run(const cv::cuda::GpuMat &bgr, bool flip);

    void buildEngineFromOnnx(const std::string &onnx_path);

    void loadEngineFromFile(const std::string &engine_path);

    void saveEngineToFile(const std::string &engine_path);

private:
    cv::Size input_size_;
    int dla_core_;
    TRTLogger gLogger_;

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    void *buffers_[2]{nullptr, nullptr};
    int input_index_ = -1;
    int output_index_ = -1;
    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;
};