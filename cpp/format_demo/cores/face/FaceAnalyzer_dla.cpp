//#error "【诊断】这是正确的FaceAnalyzer.cpp。如果编译成功，说明文件是旧的。"

#include "FaceAnalyzer_dla.hpp"
#include <iostream>
#include <stdexcept>
#include <map>
#include <numeric>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/dnn.hpp>

namespace fs = std::filesystem;

/* ------------------------------ Logger ------------------------------ */
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

/* ----------------------- 工具：自动找绑定索引 ----------------------- */
static inline int findInputBinding(const nvinfer1::ICudaEngine &engine) {
    for (int i = 0; i < engine.getNbBindings(); ++i)
        if (engine.bindingIsInput(i)) return i;
    return -1;
}

static inline int findOutputBinding(const nvinfer1::ICudaEngine &engine) {
    for (int i = 0; i < engine.getNbBindings(); ++i)
        if (!engine.bindingIsInput(i)) return i;
    return -1;
}

/* =================================================================== */
FaceAnalyzer::FaceAnalyzer(const std::string &det_model_path,
                           const std::string &rec_model_path)
        : m_det_model_path(det_model_path),
          m_rec_model_path(rec_model_path),
          m_stream(nullptr) {}

FaceAnalyzer::~FaceAnalyzer() {
    if (m_stream) cudaStreamDestroy(m_stream);
    for (void *buf: m_buffers_det) if (buf) cudaFree(buf);
    for (void *buf: m_buffers_rec) if (buf) cudaFree(buf);
}

/* ---------------------- 载入 / 构建 TensorRT 引擎 ------------------- */
static TrtUniquePtr<nvinfer1::ICudaEngine> loadOrCreateEngine(
        nvinfer1::IRuntime &runtime, const std::string &model_path,
        const std::string &provider, nvinfer1::ILogger &logger) {

    fs::path onnx_p(model_path);
    std::string engine_path = (provider == "DLA")
                              ? onnx_p.replace_extension(".dla.engine").string()
                              : onnx_p.replace_extension(".gpu.engine").string();

    /* 缓存命中直接反序列化 */
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (engine_file.good()) {
        std::cout << "[INFO] Loading pre-built engine: " << engine_path << std::endl;
        engine_file.seekg(0, std::ios::end);
        size_t size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        engine_file.read(buffer.data(), size);
        return TrtUniquePtr<nvinfer1::ICudaEngine>(
                runtime.deserializeCudaEngine(buffer.data(), size));
    }

    /* 否则从 ONNX 重新构建 */
    std::cout << "[INFO] Engine not found. Building from ONNX: "
              << model_path << std::endl;

    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    const auto explicitBatch =
            1U << static_cast<uint32_t>(
                    nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<nvinfer1::INetworkDefinition> network(
            builder->createNetworkV2(explicitBatch));
    TrtUniquePtr<nvonnxparser::IParser> parser(
            nvonnxparser::createParser(*network, logger));

    if (!parser->parseFromFile(model_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        throw std::runtime_error("Failed to parse ONNX file: " + model_path);

    TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    if (provider == "DLA") {
        std::cout << "[INFO] Configuring for DLA execution." << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(0);
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    } else {
        std::cout << "[INFO] Configuring for GPU execution." << std::endl;
        if (builder->platformHasFastFp16())
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    TrtUniquePtr<nvinfer1::IHostMemory> serialized_engine(
            builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine)
        throw std::runtime_error("Failed to build engine.");

    std::ofstream p_engine(engine_path, std::ios::binary);
    p_engine.write(reinterpret_cast<const char *>(serialized_engine->data()),
                   serialized_engine->size());
    std::cout << "[INFO] Engine built and saved to: " << engine_path << std::endl;

    return TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime.deserializeCudaEngine(serialized_engine->data(),
                                          serialized_engine->size()));
}

/* =================================================================== */
/* prepare ************************************************************ */
void FaceAnalyzer::prepare(const std::string &provider,
                           float det_thresh,
                           cv::Size det_size) {

    std::string final_provider = "CPU";
    if (provider == "CUDAExecutionProvider" || provider == "GPU")
        final_provider = "GPU";
    else if (provider == "DLA")
        final_provider = "DLA";

    std::cout << "[INFO] Preparing FaceAnalyzer with backend: "
              << final_provider << std::endl;

    m_logger = std::make_unique<TrtLogger>();
    m_runtime.reset(nvinfer1::createInferRuntime(*m_logger));

    /* --- 检测模型 --- */
    std::cout << "[INFO] Initializing detection model..." << std::endl;
    m_det_engine = loadOrCreateEngine(*m_runtime, m_det_model_path,
                                      final_provider, *m_logger);
    m_det_context.reset(m_det_engine->createExecutionContext());

    /* --- 识别模型 --- */
    std::cout << "[INFO] Initializing recognition model..." << std::endl;
    m_rec_engine = loadOrCreateEngine(*m_runtime, m_rec_model_path,
                                      final_provider, *m_logger);
    m_rec_context.reset(m_rec_engine->createExecutionContext());

    cudaStreamCreate(&m_stream);

    /* 申请 det / rec 全部绑定的 GPU buffer */
    m_buffers_det.resize(m_det_engine->getNbBindings());
    m_buffer_sizes_det.resize(m_det_engine->getNbBindings());
    for (int i = 0; i < m_det_engine->getNbBindings(); ++i) {
        auto dims = m_det_engine->getBindingDimensions(i);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) size *= dims.d[j];
        if (size == 0) continue;                  // 可变维
        m_buffer_sizes_det[i] = size * sizeof(float);
        cudaMalloc(&m_buffers_det[i], m_buffer_sizes_det[i]);
    }
    m_buffers_rec.resize(m_rec_engine->getNbBindings());
    m_buffer_sizes_rec.resize(m_rec_engine->getNbBindings());
    for (int i = 0; i < m_rec_engine->getNbBindings(); ++i) {
        auto dims = m_rec_engine->getBindingDimensions(i);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) size *= dims.d[j];
        if (size == 0) continue;
        m_buffer_sizes_rec[i] = size * sizeof(float);
        cudaMalloc(&m_buffers_rec[i], m_buffer_sizes_rec[i]);
    }

    det_thresh_ = det_thresh;
    det_size_ = det_size;
    m_is_prepared = true;
    std::cout << "[INFO] FaceAnalyzer is ready." << std::endl;
}

/* =================================================================== */
/* get_embedding_from_aligned ***************************************** */
cv::Mat FaceAnalyzer::get_embedding_from_aligned(const cv::Mat &aligned_img) {
    if (!m_is_prepared)
        throw std::runtime_error("FaceAnalyzer not prepared. Call prepare() first.");
    if (aligned_img.size() != cv::Size(112, 112))
        throw std::runtime_error("Input for get_embedding_from_aligned must be 112x112.");

    /* 1. 预处理 */
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> preprocessed(1 * 3 * 112 * 112);
    const int channel_size = 112 * 112;
    for (int h = 0; h < 112; ++h)
        for (int w = 0; w < 112; ++w) {
            cv::Vec3b px = aligned_img.at<cv::Vec3b>(h, w);
            preprocessed[0 * channel_size + h * 112 + w] = (px[2] - 127.5f) / 128.0f;
            preprocessed[1 * channel_size + h * 112 + w] = (px[1] - 127.5f) / 128.0f;
            preprocessed[2 * channel_size + h * 112 + w] = (px[0] - 127.5f) / 128.0f;
        }
    auto t1 = std::chrono::high_resolution_clock::now();

    /* 2. 找到绑定索引（自动，不再写死名字） */
    int input_idx = findInputBinding(*m_rec_engine);
    int output_idx = findOutputBinding(*m_rec_engine);
    if (input_idx < 0 || output_idx < 0)
        throw std::runtime_error("Failed to find input/output binding in rec engine.");

    /* 3. H2D Copy & 推理 */
    cudaMemcpyAsync(m_buffers_rec[input_idx], preprocessed.data(),
                    m_buffer_sizes_rec[input_idx],
                    cudaMemcpyHostToDevice, m_stream);
    m_rec_context->enqueueV2(m_buffers_rec.data(), m_stream, nullptr);

    /* 4. D2H Copy */
    cv::Mat result(1, m_buffer_sizes_rec[output_idx] / sizeof(float), CV_32F);
    cudaMemcpyAsync(result.ptr<float>(), m_buffers_rec[output_idx],
                    m_buffer_sizes_rec[output_idx],
                    cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);
    auto t2 = std::chrono::high_resolution_clock::now();

    double pre_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double infer_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "        [PERF rec] preprocess " << pre_ms
              << " ms | infer+d2h " << infer_ms << " ms" << std::endl;

    return result.clone();
}

/* =================================================================== */
/* get_embedding / get 与旧版保持一致 ********************************* */
void FaceAnalyzer::get_embedding(const cv::Mat &full_img, Face &face) {
    if (face.kps.size() != 5)
        throw std::runtime_error("Face object must have 5 keypoints for alignment.");

    auto align_start = std::chrono::high_resolution_clock::now();
    const std::vector<cv::Point2f> dst_pts = {{38.2946f, 51.6963f},
                                              {73.5318f, 51.5014f},
                                              {56.0252f, 71.7366f},
                                              {41.5493f, 92.3655f},
                                              {70.7299f, 92.2041f}};
    cv::Mat M = cv::estimateAffinePartial2D(face.kps, dst_pts);
    if (M.empty()) {
        face.embedding = cv::Mat();
        return;
    }
    cv::Mat aligned;
    cv::warpAffine(full_img, aligned, M, cv::Size(112, 112));
    face.aligned_face = aligned.clone();
    auto align_end = std::chrono::high_resolution_clock::now();

    auto infer_start = std::chrono::high_resolution_clock::now();
    face.embedding = get_embedding_from_aligned(aligned);
    auto infer_end = std::chrono::high_resolution_clock::now();

    double align_ms = std::chrono::duration<double, std::milli>(align_end - align_start).count();
    double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    double total_ms = align_ms + infer_ms;
    if (total_ms > 1.0)
        std::cout << "      [PERF get_emb] total " << total_ms
                  << " ms | align " << align_ms
                  << " ms | infer " << infer_ms << " ms\n";
}

std::vector<Face> FaceAnalyzer::get(const cv::Mat &img) {
    if (!m_is_prepared) {
        std::cerr << "[ERROR] FaceAnalyzer::get called before prepare().\n";
        return {};
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto faces = detect(img);
    auto t1 = std::chrono::high_resolution_clock::now();

    double det_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double rec_ms_total = 0.0;
    for (auto &f: faces) {
        auto tr0 = std::chrono::high_resolution_clock::now();
        get_embedding(img, f);
        auto tr1 = std::chrono::high_resolution_clock::now();
        rec_ms_total += std::chrono::duration<double, std::milli>(tr1 - tr0).count();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();

    std::cout << "[FaceAnalyzer::get] total " << total_ms << " ms (det "
              << det_ms << " ms, rec_all " << rec_ms_total << " ms) for "
              << faces.size() << " faces.\n";
    return faces;
}

/* =================================================================== */
/* detect ************************************************************* */
std::vector<Face> FaceAnalyzer::detect(const cv::Mat &img) {
    if (!m_is_prepared) throw std::runtime_error("FaceAnalyzer not prepared.");
    if (img.empty()) return {};

    /* ---------- 1. 缩放 / pad 到 det_size_ ---------- */
    float im_ratio = static_cast<float>(img.rows) / img.cols;
    float mdl_ratio = static_cast<float>(det_size_.height) / det_size_.width;
    int new_w, new_h;
    if (im_ratio > mdl_ratio) {       // 高比较长
        new_h = det_size_.height;
        new_w = static_cast<int>(new_h / im_ratio);
    } else {                          // 宽比较长
        new_w = det_size_.width;
        new_h = static_cast<int>(new_w * im_ratio);
    }
    float scale = static_cast<float>(new_h) / img.rows;
    if (scale == 0) scale = 1.f;

    cv::Mat resized, det_img = cv::Mat::zeros(det_size_, CV_8UC3);
    cv::resize(img, resized, cv::Size(new_w, new_h));
    resized.copyTo(det_img(cv::Rect(0, 0, new_w, new_h)));

    /* ---------- 2. HWC(BGR) -> CHW(RGB) & 归一化 ---------- */
    std::vector<float> blob(1 * 3 * det_size_.height * det_size_.width);
    int H = det_size_.height, W = det_size_.width, ch_size = H * W;
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            cv::Vec3b p = det_img.at<cv::Vec3b>(h, w);
            blob[0 * ch_size + h * W + w] = (p[2] - 127.5f) / 128.0f;
            blob[1 * ch_size + h * W + w] = (p[1] - 127.5f) / 128.0f;
            blob[2 * ch_size + h * W + w] = (p[0] - 127.5f) / 128.0f;
        }

    /* ---------- 3. 推理 ---------- */
    int input_idx = findInputBinding(*m_det_engine);
    if (input_idx < 0) throw std::runtime_error("Detect engine input binding not found.");

    cudaMemcpyAsync(m_buffers_det[input_idx], blob.data(),
                    m_buffer_sizes_det[input_idx],
                    cudaMemcpyHostToDevice, m_stream);
    m_det_context->enqueueV2(m_buffers_det.data(), m_stream, nullptr);

    /* ---------- 4. 拷贝输出 ---------- */
    std::vector<cv::Mat> outs;
    for (const auto &name: m_det_output_names) {
        int idx = m_det_engine->getBindingIndex(name.c_str());
        if (idx < 0) {
            std::cerr << "[WARN] Detection output binding not found: "
                      << name << std::endl;
            continue;
        }
        cv::Mat out(1, m_buffer_sizes_det[idx] / sizeof(float), CV_32F);
        cudaMemcpyAsync(out.ptr<float>(), m_buffers_det[idx],
                        m_buffer_sizes_det[idx],
                        cudaMemcpyDeviceToHost, m_stream);
        outs.push_back(out);
    }
    cudaStreamSynchronize(m_stream);

    if (outs.size() != m_det_output_names.size()) {
        std::cerr << "[WARN] Detect output num mismatch, abort detection.\n";
        return {};
    }

    /* ---------- 5. 后处理 (根据 SCRFD 原理) ---------- */
    std::vector<cv::Rect2d> bboxes;
    std::vector<float> scores;
    std::vector<std::vector<cv::Point2f>> all_kps;
    std::vector<int> strides = {8, 16, 32};          // 与模型保持一致
    for (size_t i = 0; i < strides.size(); ++i) {
        const int stride = strides[i];
        const int h_out = det_size_.height / stride;
        const int w_out = det_size_.width / stride;

        const float *score_data = outs[i].ptr<float>();
        const float *bbox_data = outs[i + strides.size()].ptr<float>();
        const float *kps_data = outs[i + 2 * strides.size()].ptr<float>();

        for (int y = 0; y < h_out; ++y) {
            for (int x = 0; x < w_out; ++x) {
                for (int a = 0; a < 2; ++a) {       // 每格两个 anchor
                    int idx = (y * w_out + x) * 2 + a;
                    float sc = score_data[idx];
                    if (sc < det_thresh_) continue;

                    const float *bb = &bbox_data[idx * 4];
                    float x1 = (x + 0.5f - bb[0]) * stride / scale;
                    float y1 = (y + 0.5f - bb[1]) * stride / scale;
                    float x2 = (x + 0.5f + bb[2]) * stride / scale;
                    float y2 = (y + 0.5f + bb[3]) * stride / scale;

                    bboxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
                    scores.push_back(sc);

                    std::vector<cv::Point2f> kps;
                    const float *kp_ptr = &kps_data[idx * 10];
                    for (int k = 0; k < 5; ++k) {
                        float kx = (x + 0.5f + kp_ptr[k * 2]) * stride / scale;
                        float ky = (y + 0.5f + kp_ptr[k * 2 + 1]) * stride / scale;
                        kps.emplace_back(kx, ky);
                    }
                    all_kps.push_back(kps);
                }
            }
        }
    }

    /* NMS */
    std::vector<int> keep;
    cv::dnn::NMSBoxes(bboxes, scores, det_thresh_, 0.4, keep);

    /* ---------- 6. 组装结果 ---------- */
    std::vector<Face> faces;
    for (int id: keep) {
        Face f;
        f.bbox = bboxes[id];
        f.det_score = scores[id];
        f.kps = all_kps[id];
        faces.push_back(f);
    }
    return faces;
}