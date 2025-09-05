#include "PersonReid_dla.hpp"
#include <opencv2/imgproc.hpp>
#include <cuda_runtime_api.h>
#include <numeric>
#include <fstream>
#include <iostream>
#include <vector>

// --- CUDA 错误检查宏 ---
#define CHECK_CUDA(status)                                                    \
    do {                                                                      \
        auto ret = (status);                                                  \
        if (ret != 0) {                                                       \
            std::cerr << "CUDA Error: " << cudaGetErrorString((cudaError_t)ret) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

// --- TensorRT 日志记录器实现 ---
void TRTLogger::log(Severity severity, const char *msg) noexcept {
    // 只打印 Warning 及以上级别的日志，避免过多 INFO 输出
    if (severity <= Severity::kWARNING) {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

// --- 工具函数：根据维度计算元素总数 ---
static inline size_t sizeFromDims(const nvinfer1::Dims &d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

// --- 构造函数与析构函数 ---

PersonReidDLA::PersonReidDLA(const std::string &onnx_path,
                             int in_w, int in_h,
                             int dla_core,
                             const std::string &engine_cache)
        : input_size_(in_w, in_h),
          dla_core_(dla_core) {

    runtime_.reset(nvinfer1::createInferRuntime(gLogger_));
    if (!runtime_) throw std::runtime_error("Failed to create TensorRT Runtime.");

    /* 新增 ↓↓↓ 让 Runtime 在反序列化/执行阶段绑定到期望的 DLA Core */
    runtime_->setDLACore(dla_core_);
    /* 新增 ↑↑↑ */

    // 检查是否有缓存的引擎文件
    if (!engine_cache.empty() && std::ifstream(engine_cache).good()) {
        std::cout << "Loading TRT engine from cache: " << engine_cache << std::endl;
        loadEngineFromFile(engine_cache);
    } else {
        std::cout << "Building TRT engine from ONNX: " << onnx_path << std::endl;
        buildEngineFromOnnx(onnx_path);
        if (!engine_cache.empty()) {
            saveEngineToFile(engine_cache);
            std::cout << "TRT engine cached to: " << engine_cache << std::endl;
        }
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Failed to create TensorRT Execution Context.");

    // 获取输入输出节点的索引 (注意，这里的 "input" 和 "output" 需要和你的 ONNX 模型一致)
    input_index_ = engine_->getBindingIndex("input");
    output_index_ = engine_->getBindingIndex("output");
    if (input_index_ < 0 || output_index_ < 0) {
        throw std::runtime_error("Invalid ONNX input/output names. Expected 'input' and 'output'.");
    }

    // 计算输入输出缓冲区大小并分配 CUDA 内存
    input_bytes_ = sizeFromDims(engine_->getBindingDimensions(input_index_)) * sizeof(float);
    output_bytes_ = sizeFromDims(engine_->getBindingDimensions(output_index_)) * sizeof(float);

    CHECK_CUDA(cudaMalloc(&buffers_[input_index_], input_bytes_));
    CHECK_CUDA(cudaMalloc(&buffers_[output_index_], output_bytes_));
}

PersonReidDLA::~PersonReidDLA() {
    if (buffers_[input_index_]) cudaFree(buffers_[input_index_]);
    if (buffers_[output_index_]) cudaFree(buffers_[output_index_]);
}

// --- 公共接口实现 ---

cv::Mat PersonReidDLA::extract_feat(const cv::Mat &bgr) {
    // 分别对原图和翻转图进行推理
    cv::Mat feat_orig = run(bgr, false);
    cv::Mat feat_flipped = run(bgr, true);

    if (feat_orig.empty() || feat_flipped.empty()) {
        std::cerr << "Warning: Inference returned an empty feature vector." << std::endl;
        return cv::Mat();
    }
    // 特征融合
    cv::Mat feat_sum = feat_orig + feat_flipped;

    // L2 归一化
    cv::Mat feat_norm;
    cv::normalize(feat_sum, feat_norm, 1.0, 0.0, cv::NORM_L2);

    return feat_norm;
}

// --- 私有方法实现 ---

cv::Mat PersonReidDLA::run(const cv::Mat &bgr, bool flip) {
    // 1. 图像预处理
    cv::Mat img_flipped;
    const cv::Mat &img_to_process = flip ? (cv::flip(bgr, img_flipped, 1), img_flipped) : bgr;

    cv::Mat img_resized;
    cv::resize(img_to_process, img_resized, input_size_);

    cv::Mat img_rgb;
    cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGR2RGB);

    cv::Mat img_float;
    img_rgb.convertTo(img_float, CV_32F, 1.0 / 255.0);

    // 标准化
    std::vector<cv::Mat> channels(3);
    cv::split(img_float, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f; // R
    channels[1] = (channels[1] - 0.456f) / 0.224f; // G
    channels[2] = (channels[2] - 0.406f) / 0.225f; // B
    cv::Mat normalized;
    cv::merge(channels, normalized);

    // HWC -> NCHW 格式转换 (OpenCV Mat是HWC，TensorRT需要CHW)
    std::vector<float> input_data(input_size_.width * input_size_.height * 3);
    // CHW排布
    float *ptr_r = input_data.data();
    float *ptr_g = ptr_r + input_size_.width * input_size_.height;
    float *ptr_b = ptr_g + input_size_.width * input_size_.height;
    for (int i = 0; i < input_size_.height; ++i) {
        for (int j = 0; j < input_size_.width; ++j) {
            ptr_r[i * input_size_.width + j] = normalized.at<cv::Vec3f>(i, j)[0];
            ptr_g[i * input_size_.width + j] = normalized.at<cv::Vec3f>(i, j)[1];
            ptr_b[i * input_size_.width + j] = normalized.at<cv::Vec3f>(i, j)[2];
        }
    }

    // 2. 数据拷贝到 GPU
    CHECK_CUDA(cudaMemcpy(buffers_[input_index_], input_data.data(), input_bytes_, cudaMemcpyHostToDevice));

    // 3. 执行推理
    context_->enqueueV2(buffers_, 0, nullptr);

    // 4. 数据从 GPU 拷回 CPU
    std::vector<float> output_data(output_bytes_ / sizeof(float));
    CHECK_CUDA(cudaMemcpy(output_data.data(), buffers_[output_index_], output_bytes_, cudaMemcpyDeviceToHost));

    // 5. 结果封装成 cv::Mat
    return cv::Mat(1, (int) output_data.size(), CV_32F, output_data.data()).clone();
}

void PersonReidDLA::buildEngineFromOnnx(const std::string &onnx_path) {
    using namespace nvinfer1;
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger_));
    if (!builder) throw std::runtime_error("Failed to create TRT Builder.");

    auto network = std::unique_ptr<INetworkDefinition>(
            builder->createNetworkV2(1U << (int) NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    if (!network) throw std::runtime_error("Failed to create TRT Network.");

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config) throw std::runtime_error("Failed to create TRT Builder Config.");

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger_));
    if (!parser) throw std::runtime_error("Failed to create ONNX Parser.");

    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX file: " + onnx_path);
    }

    // DLA 配置
    config->setMaxWorkspaceSize(1ULL << 28); // 256MB
    config->setDefaultDeviceType(DeviceType::kDLA);
    config->setDLACore(dla_core_);
    config->setFlag(BuilderFlag::kFP16); // DLA 必须使用 FP16 或 INT8
    config->setFlag(BuilderFlag::kGPU_FALLBACK); // <-- 新增此行以启用 GPU Fallback

    engine_.reset(builder->buildEngineWithConfig(*network, *config));
    if (!engine_) throw std::runtime_error("Failed to build TRT engine.");
}

void PersonReidDLA::loadEngineFromFile(const std::string &engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) throw std::runtime_error("Failed to open engine file: " + engine_path);

    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!engine_) throw std::runtime_error("Failed to deserialize TRT engine.");
}

void PersonReidDLA::saveEngineToFile(const std::string &engine_path) {
    std::ofstream file(engine_path, std::ios::binary);
    if (!file.good()) throw std::runtime_error("Failed to open engine file for writing: " + engine_path);

    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(engine_->serialize());
    if (!serialized_engine) throw std::runtime_error("Failed to serialize TRT engine.");

    file.write(static_cast<const char *>(serialized_engine->data()), serialized_engine->size());
}