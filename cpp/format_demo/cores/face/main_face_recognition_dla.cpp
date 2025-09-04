#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <filesystem>
#include <algorithm> // 需要此头文件用于排序
#include <memory>    // 用于 std::unique_ptr

#include <opencv2/opencv.hpp>
// #include <opencv2/dnn.hpp> // 不再需要

// ======================= 【新增】TensorRT 相关头文件 =======================
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

// 使用 `std::filesystem` 需要此别名
namespace fs = std::filesystem;

// ======================= 【新增】TensorRT Logger =======================
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        // 只打印 Error 和 Warning 级别的信息，避免过多输出
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

// ======================= 【新增】智能指针包装器 =======================
// 让 std::unique_ptr 可以管理 TensorRT 的对象
template<typename T>
struct TrtDeleter {
    void operator()(T *obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

template<typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

// ======================= 【新增】辅助函数：生成 engine 文件路径 =======================
std::string get_engine_path(const std::string &onnx_path) {
    return fs::path(onnx_path).replace_extension(".dla.engine").string();
}

// ======================= 【新增】辅助函数：从 ONNX 构建 DLA 引擎 =======================
TrtUniquePtr<nvinfer1::ICudaEngine> build_and_save_engine(const std::string &onnx_path, Logger &logger) {
    std::string engine_path = get_engine_path(onnx_path);
    std::cout << "[INFO] Building engine from " << onnx_path << " for DLA..." << std::endl;
    std::cout << "[INFO] This may take a few minutes..." << std::endl;

    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
    TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));

    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX file: " + onnx_path);
    }

    TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(1 << 30); // 1GB workspace size

    // --- 配置为 DLA ---
    config->setFlag(nvinfer1::BuilderFlag::kFP16); // DLA 通常需要 FP16 或 INT8 模式
    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    config->setDLACore(0);                         // 使用 DLA 核心 0
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);   // <-- 新增，允许不支持层自动回退到 GPU

    TrtUniquePtr<nvinfer1::IHostMemory> serialized_engine(builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine) {
        throw std::runtime_error("Failed to build engine.");
    }

    // 保存到文件
    std::ofstream p_engine(engine_path, std::ios::binary);
    if (!p_engine) {
        throw std::runtime_error("Failed to open engine file for writing: " + engine_path);
    }
    p_engine.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());
    p_engine.close();
    std::cout << "[INFO] Engine built and saved to: " << engine_path << std::endl;

    // 从序列化的数据中创建引擎返回
    TrtUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    return TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));
}

int main() {
    std::cout << "\n\n!!!!!!!!!! NEWEST CODE IS RUNNING, BUILD IS CLEAN !!!!!!!!!!\n\n" << std::endl;

    // ======================= Configuration =======================
    // 1. 设置包含BMP文件的文件夹路径
    std::string bmp_folder_path = "/mnt/nfs/face_aligned_py_bmp/";

    // 2. 设置统一的输出TXT文件的路径
    std::string output_txt_path = "/mnt/nfs/embeddings_cpp_from_aligned_bmps.txt";

    // 3. 设置识别（recognition）模型的路径
    std::string rec_model_path = "/mnt/nfs/w600k_r50_simplified.onnx";
    // =============================================================

    std::cout << "[INFO] Starting batch feature extraction test..." << std::endl;
    std::cout << "[INFO] Reading BMP images from: " << bmp_folder_path << std::endl;

    // ======================= 【修改的部分在此】 =======================
    // --- 1. 初始化 TensorRT Logger, Runtime 和 Engine ---
    Logger logger;
    TrtUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;

    std::string engine_path = get_engine_path(rec_model_path);
    std::ifstream engine_file(engine_path, std::ios::binary);

    if (engine_file.good()) {
        std::cout << "[INFO] Loading pre-built DLA engine from: " << engine_path << std::endl;
        engine_file.seekg(0, std::ios::end);
        size_t size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        engine_file.read(buffer.data(), size);
        engine.reset(runtime->deserializeCudaEngine(buffer.data(), size));
    } else {
        engine = build_and_save_engine(rec_model_path, logger);
    }

    if (!engine) {
        throw std::runtime_error("Failed to create TensorRT engine.");
    }

    TrtUniquePtr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("Failed to create execution context.");
    }

    // --- 2. 准备 TRT 的输入输出 buffer ---
    // 动态查找输入和输出绑定索引，而不是硬编码名称
    int input_idx = -1;
    int output_idx = -1;
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        if (engine->bindingIsInput(i)) {
            if (input_idx == -1) { // 找到第一个输入
                input_idx = i;
            }
        } else {
            if (output_idx == -1) { // 找到第一个输出
                output_idx = i;
            }
        }
    }

    if (input_idx == -1 || output_idx == -1) {
        throw std::runtime_error("Failed to find input or output binding in the engine.");
    }

    // 使用引擎中的绑定数量来安全地调整 buffer 数组大小
    std::vector<void *> buffers(engine->getNbBindings());

    auto input_dims = engine->getBindingDimensions(input_idx);
    size_t input_size = 1;
    for (int j = 0; j < input_dims.nbDims; ++j) { input_size *= input_dims.d[j]; }

    auto output_dims = engine->getBindingDimensions(output_idx);
    size_t output_size = 1;
    for (int j = 0; j < output_dims.nbDims; ++j) { output_size *= output_dims.d[j]; }

    // 在GPU上分配内存
    cudaMalloc(&buffers[input_idx], input_size * sizeof(float));
    cudaMalloc(&buffers[output_idx], output_size * sizeof(float));

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 用于预处理的CPU端buffer
    std::vector<float> preprocessed_input(input_size);
    std::cout << "[INFO] TensorRT engine and buffers ready for DLA inference." << std::endl;
    // ======================= 【修改结束】 =======================

    try {
        // --- 3. 准备输出文件 (打开一次，准备写入) ---
        std::ofstream out_file(output_txt_path);
        if (!out_file.is_open()) {
            throw std::runtime_error("Failed to open output file for writing: " + output_txt_path);
        }

        // --- 4. 收集并排序所有BMP文件的路径 ---
        std::vector<fs::path> bmp_paths;
        if (!fs::exists(bmp_folder_path) || !fs::is_directory(bmp_folder_path)) {
            throw std::runtime_error("Input folder does not exist or is not a directory: " + bmp_folder_path);
        }
        for (const auto &entry: fs::directory_iterator(bmp_folder_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bmp") {
                bmp_paths.push_back(entry.path());
            }
        }
        std::sort(bmp_paths.begin(), bmp_paths.end()); // 确保处理顺序一致
        std::cout << "[INFO] Found " << bmp_paths.size() << " BMP files to process." << std::endl;

        // --- 5. 遍历所有BMP文件进行处理 ---
        for (const auto &bmp_path: bmp_paths) {

            // --- 读取图像 ---
            cv::Mat aligned_img = cv::imread(bmp_path.string());
            if (aligned_img.empty() || aligned_img.size() != cv::Size(112, 112)) {
                std::cerr << "[WARNING] Failed to read or invalid image size: " << bmp_path << std::endl;
                continue; // 跳过这个文件，继续处理下一个
            }

            // ======================= 【修改的部分在此】 =======================
            // --- 手动图像预处理 (HWC->CHW, BGR->RGB, Normalize) ---
            int C = input_dims.d[1];
            int H = input_dims.d[2];
            int W = input_dims.d[3];
            int channel_size = H * W;

            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    cv::Vec3b pixel = aligned_img.at<cv::Vec3b>(h, w);
                    preprocessed_input[0 * channel_size + h * W + w] = (pixel[2] - 127.5f) / 128.0f; // R
                    preprocessed_input[1 * channel_size + h * W + w] = (pixel[1] - 127.5f) / 128.0f; // G
                    preprocessed_input[2 * channel_size + h * W + w] = (pixel[0] - 127.5f) / 128.0f; // B
                }
            }

            // --- 执行推理 ---
            // 1. 将预处理好的数据从CPU拷贝到GPU
            cudaMemcpyAsync(buffers[input_idx], preprocessed_input.data(), input_size * sizeof(float),
                            cudaMemcpyHostToDevice, stream);

            // 2. 连续执行推理 n 次
            for (int infer_iter = 0; infer_iter < 1000; ++infer_iter) {
                context->enqueueV2(buffers.data(), stream, nullptr);
            }

            // 3. 将结果从GPU拷贝回CPU（取最后一次推理结果即可）
            cv::Mat embedding_raw(1, output_size, CV_32F);
            cudaMemcpyAsync(embedding_raw.ptr<float>(), buffers[output_idx], output_size * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);

            // 4. 等待所有任务完成
            cudaStreamSynchronize(stream);

            if (embedding_raw.empty()) {
                std::cerr << "[WARNING] Failed to get embedding for: " << bmp_path << std::endl;
                continue;
            }
            // ======================= 【修改结束】 =======================

            // --- L2归一化 ---
            cv::Mat embedding_normalized;
            cv::normalize(embedding_raw, embedding_normalized, 1.0, 0.0, cv::NORM_L2);

            // --- 写入文件 ---
            std::string face_id = bmp_path.stem().string();
            out_file << face_id;

            const float *data = embedding_normalized.ptr<float>(0);
            for (int i = 0; i < embedding_normalized.cols; ++i) {
                out_file << "," << std::fixed << std::setprecision(8) << data[i];
            }
            out_file << "\n";

            // 打印进度
            std::cout << "Processed: " << face_id << std::endl;
        }

        // --- 6. 释放资源 ---
        out_file.close();
        cudaStreamDestroy(stream);
        cudaFree(buffers[input_idx]);
        cudaFree(buffers[output_idx]);

        std::cout << "[SUCCESS] All embeddings saved to " << output_txt_path << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "[ERROR] An exception occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}