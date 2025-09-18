#include "PersonReid_dla.hpp"
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
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
    if (severity <= Severity::kWARNING) {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

// --- 工具函数：计算张量大小 ---
static inline size_t sizeFromDims(const nvinfer1::Dims &d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

// --- 构造与析构 ---
PersonReidDLA::PersonReidDLA(const std::string &onnx_path,
                             int in_w, int in_h,
                             int dla_core,
                             const std::string &engine_cache)
        : input_size_(in_w, in_h),
          dla_core_(dla_core) {

    runtime_.reset(nvinfer1::createInferRuntime(gLogger_));
    if (!runtime_) throw std::runtime_error("Failed to create TensorRT Runtime.");

    runtime_->setDLACore(dla_core_);

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

    input_index_ = engine_->getBindingIndex("input");
    output_index_ = engine_->getBindingIndex("output");
    if (input_index_ < 0 || output_index_ < 0) {
        throw std::runtime_error("Invalid ONNX input/output names. Expected 'input' and 'output'.");
    }

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
cv::cuda::GpuMat PersonReidDLA::extract_feat(const cv::cuda::GpuMat &rgb) {
    cv::cuda::GpuMat feat_orig = run(rgb, false);
    cv::cuda::GpuMat feat_flip = run(rgb, true);

    if (feat_orig.empty() || feat_flip.empty()) {
        std::cerr << "Warning: empty feature!" << std::endl;
        return cv::cuda::GpuMat();
    }

    cv::cuda::GpuMat feat_sum;
    cv::cuda::add(feat_orig, feat_flip, feat_sum);

    // L2 归一化
    cv::cuda::GpuMat feat_sq;
    cv::cuda::multiply(feat_sum, feat_sum, feat_sq);
    cv::Scalar sum_scalar = cv::cuda::sum(feat_sq);
    float norm_val = std::sqrt(sum_scalar[0] + 1e-12f);

    cv::cuda::GpuMat feat_norm;
    cv::cuda::multiply(feat_sum, cv::Scalar(1.0f / norm_val), feat_norm);

    return feat_norm;
}

// --- 私有方法 ---
cv::cuda::GpuMat PersonReidDLA::run(const cv::cuda::GpuMat &rgb, bool flip) {
    // 1. 图像预处理 (GPU 完成)
    cv::cuda::GpuMat img_proc;
    if (flip) cv::cuda::flip(rgb, img_proc, 1);
    else img_proc = rgb;

    cv::cuda::GpuMat img_resized, img_float;
    cv::cuda::resize(img_proc, img_resized, input_size_);
    img_resized.convertTo(img_float, CV_32F, 1.0 / 255.0);

    // 标准化
    std::vector<cv::cuda::GpuMat> chans(3);
    cv::cuda::split(img_float, chans);

    cv::cuda::subtract(chans[0], cv::Scalar(0.485f), chans[0]);
    cv::cuda::divide(chans[0], cv::Scalar(0.229f), chans[0]);

    cv::cuda::subtract(chans[1], cv::Scalar(0.456f), chans[1]);
    cv::cuda::divide(chans[1], cv::Scalar(0.224f), chans[1]);

    cv::cuda::subtract(chans[2], cv::Scalar(0.406f), chans[2]);
    cv::cuda::divide(chans[2], cv::Scalar(0.225f), chans[2]);

    // 将归一化后的数据拷贝到输入 buffer (CHW 格式)
    for (int c = 0; c < 3; c++) {
        cv::cuda::GpuMat channel(input_size_, CV_32F,
                                 (float *) buffers_[input_index_] + c * input_size_.area());
        chans[c].copyTo(channel);
    }

    // 2. TensorRT 执行推理
    context_->enqueueV2(buffers_, 0, nullptr);

    // 3. 输出 buffer 封装为 GpuMat
    cv::cuda::GpuMat out(1, output_bytes_ / sizeof(float), CV_32F, buffers_[output_index_]);
    return out.clone(); // clone 防止下次推理覆盖
}

// --- Engine 构建/加载/保存 ---
void PersonReidDLA::buildEngineFromOnnx(const std::string &onnx_path) {
    using namespace nvinfer1;
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger_));
    if (!builder) throw std::runtime_error("Failed to create TRT Builder.");

    auto network = std::unique_ptr<INetworkDefinition>(
            builder->createNetworkV2(1U << (int) NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    if (!network) throw std::runtime_error("Failed to create TRT Network.");

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config) throw std::runtime_error("Failed to create TRT Builder Config.");

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, gLogger_));
    if (!parser) throw std::runtime_error("Failed to create ONNX Parser.");

    if (!parser->parseFromFile(onnx_path.c_str(), int(ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX: " + onnx_path);
    }

    config->setMaxWorkspaceSize(1ULL << 28);
    config->setDefaultDeviceType(DeviceType::kDLA);
    config->setDLACore(dla_core_);
    config->setFlag(BuilderFlag::kFP16);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    engine_.reset(builder->buildEngineWithConfig(*network, *config));
    if (!engine_) throw std::runtime_error("Failed to build TRT engine.");
}

void PersonReidDLA::loadEngineFromFile(const std::string &engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) throw std::runtime_error("Failed to open engine file: " + engine_path);

    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), {});
    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!engine_) throw std::runtime_error("Failed to deserialize engine.");
}

void PersonReidDLA::saveEngineToFile(const std::string &engine_path) {
    std::ofstream file(engine_path, std::ios::binary);
    if (!file.good()) throw std::runtime_error("Failed to open file for writing: " + engine_path);

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(engine_->serialize());
    if (!serialized) throw std::runtime_error("Failed to serialize engine.");

    file.write((const char *) serialized->data(), serialized->size());
}