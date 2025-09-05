#pragma once
/*
 *  Jetson-DLA 版本的 ReID 推理器（基于 TensorRT 原生 API）
 *  - 支持 ONNX->TRT 引擎自动构建并缓存
 *  - 支持 DLA0 / DLA1 选择
 *  - extract_feat() 的行为与原先 PersonReid 类（使用OpenCV DNN）保持一致
 */
#include <opencv2/core.hpp>
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
    /**
     * @brief 构造函数：初始化 DLA 引擎。
     * @param onnx_path   ONNX 模型的路径。
     * @param input_w     模型输入的宽度 (e.g., 128)。
     * @param input_h     模型输入的高度 (e.g., 256)。
     * @param dla_core    要使用的 DLA核心 (0 或 1)。默认为 0。
     * @param engine_path （可选）用于缓存 TensorRT 引擎的路径。如果文件存在，则直接加载；否则，构建后保存于此。如果为空，则每次都重新构建引擎。
     */
    PersonReidDLA(const std::string &onnx_path,
                  int input_w,
                  int input_h,
                  int dla_core = 0,
                  const std::string &engine_path = "");

    /**
     * @brief 析构函数：释放 CUDA 内存。
     */
    ~PersonReidDLA();

    /**
     * @brief 提取特征向量，逻辑与原版完全一致。
     * @param bgr 从 cv::imread 加载的 BGR 格式图像。
     * @return cv::Mat 1xN 的浮点型特征向量 (CV_32F)，经过 L2 归一化。
     */
    cv::Mat extract_feat(const cv::Mat &bgr);

private:
    /**
     * @brief 内部函数：对单张图像（可选择翻转）执行一次完整的预处理和推理流程。
     * @param bgr 输入图像。
     * @param flip 是否水平翻转。
     * @return cv::Mat 未经归一化的原始特征向量。
     */
    cv::Mat run(const cv::Mat &bgr, bool flip);

    // --- TensorRT 引擎管理 ---
    void buildEngineFromOnnx(const std::string &onnx_path);

    void loadEngineFromFile(const std::string &engine_path);

    void saveEngineToFile(const std::string &engine_path);

private:
    cv::Size input_size_;
    int dla_core_;
    TRTLogger gLogger_; // Logger 实例

    // --- TensorRT 核心组件 ---
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // --- CUDA 内存缓冲区 ---
    void *buffers_[2]{nullptr, nullptr};
    int input_index_ = -1;
    int output_index_ = -1;
    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;
};