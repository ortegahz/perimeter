#include "FaceAnalyzer_dla.hpp"
#include <iostream>
#include <stdexcept>
#include <map>
#include <numeric>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <functional>
#include <cmath>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

namespace fs = std::filesystem;

// [1] =========== 工具和后处理 ===========

class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
};

static inline int findInputBinding(const nvinfer1::ICudaEngine &engine, const char *name = nullptr) {
    if (name) return engine.getBindingIndex(name);
    for (int i = 0; i < engine.getNbBindings(); ++i)
        if (engine.bindingIsInput(i)) return i;
    return -1;
}

static inline int findOutputBinding(const nvinfer1::ICudaEngine &engine) {
    for (int i = 0; i < engine.getNbBindings(); ++i)
        if (!engine.bindingIsInput(i)) return i;
    return -1;
}

struct ModelConfig {
    std::string name = "mobilenet0.25";
    std::vector<std::vector<int>> min_sizes = {{16, 32},
                                               {64, 128},
                                               {256, 512}};
    std::vector<int> steps = {8, 16, 32};
    std::vector<float> variance = {0.1f, 0.2f};
    bool clip = false;
};

std::vector<BBox> generate_priors(int im_height, int im_width) {
    ModelConfig cfg;
    std::vector<std::pair<int, int>> feature_maps;
    for (const auto &step: cfg.steps)
        feature_maps.push_back({(int) ceil((float) im_height / step), (int) ceil((float) im_width / step)});
    std::vector<BBox> anchors;
    for (size_t k = 0; k < feature_maps.size(); ++k) {
        auto f = feature_maps[k];
        auto min_sizes_k = cfg.min_sizes[k];
        for (int i = 0; i < f.first; ++i)
            for (int j = 0; j < f.second; ++j)
                for (const auto &min_size: min_sizes_k) {
                    float s_kx = (float) min_size / im_width;
                    float s_ky = (float) min_size / im_height;
                    float cx = (j + 0.5f) * cfg.steps[k] / im_width;
                    float cy = (i + 0.5f) * cfg.steps[k] / im_height;
                    anchors.push_back({cx, cy, s_kx, s_ky});
                }
    }
    return anchors;
}

std::vector<BBox> decode_boxes(const std::vector<float> &loc, const std::vector<BBox> &priors) {
    ModelConfig cfg;
    std::vector<BBox> boxes;
    boxes.reserve(priors.size());
    for (size_t i = 0; i < priors.size(); ++i) {
        float p_cx = priors[i].x1, p_cy = priors[i].y1, p_w = priors[i].x2, p_h = priors[i].y2;
        float loc_x = loc[i * 4 + 0], loc_y = loc[i * 4 + 1], loc_w = loc[i * 4 + 2], loc_h = loc[i * 4 + 3];
        float cx = p_cx + loc_x * cfg.variance[0] * p_w, cy = p_cy + loc_y * cfg.variance[0] * p_h;
        float w = p_w * exp(loc_w * cfg.variance[1]), h = p_h * exp(loc_h * cfg.variance[1]);
        boxes.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2});
    }
    return boxes;
}

std::vector<Landmark> decode_landmarks(const std::vector<float> &pre, const std::vector<BBox> &priors) {
    ModelConfig cfg;
    std::vector<Landmark> landmarks;
    landmarks.reserve(priors.size());
    for (size_t i = 0; i < priors.size(); ++i) {
        float p_cx = priors[i].x1, p_cy = priors[i].y1, p_w = priors[i].x2, p_h = priors[i].y2;
        Landmark l;
        for (int j = 0; j < 5; ++j) {
            l.x_coords[j] = p_cx + pre[i * 10 + j * 2] * cfg.variance[0] * p_w;
            l.y_coords[j] = p_cy + pre[i * 10 + j * 2 + 1] * cfg.variance[0] * p_h;
        }
        landmarks.push_back(l);
    }
    return landmarks;
}

std::vector<int> cpu_nms(std::vector<Detection> &dets, float thresh) {
    std::vector<int> keep;
    if (dets.empty()) return keep;
    std::vector<size_t> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t i, size_t j) { return dets[i].score > dets[j].score; });
    std::vector<float> areas(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        auto &d = dets[i].box;
        areas[i] = (d.x2 - d.x1 + 1) * (d.y2 - d.y1 + 1);
    }
    while (!order.empty()) {
        size_t i = order[0];
        keep.push_back(i);
        std::vector<size_t> new_order;
        for (size_t j = 1; j < order.size(); ++j) {
            size_t current_idx = order[j];
            auto &box_i = dets[i].box;
            auto &box_j = dets[current_idx].box;
            float xx1 = std::max(box_i.x1, box_j.x1), yy1 = std::max(box_i.y1, box_j.y1);
            float xx2 = std::min(box_i.x2, box_j.x2), yy2 = std::min(box_i.y2, box_j.y2);
            float w = std::max(0.0f, xx2 - xx1 + 1), h = std::max(0.0f, yy2 - yy1 + 1), inter = w * h;
            float ovr = inter / (areas[i] + areas[current_idx] - inter);
            if (ovr <= thresh) new_order.push_back(current_idx);
        }
        order = new_order;
    }
    return keep;
}

// [2] ========= FaceAnalyzer实现部分 ===========

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

static TrtUniquePtr<nvinfer1::ICudaEngine> loadOrCreateEngine(
        nvinfer1::IRuntime &runtime, const std::string &model_path,
        const std::string &provider, nvinfer1::ILogger &logger) {
    fs::path onnx_p(model_path);
    std::string engine_path = (provider == "DLA")
                              ? onnx_p.replace_extension(".dla.engine").string()
                              : onnx_p.replace_extension(".gpu.engine").string();

    std::ifstream engine_file(engine_path, std::ios::binary);
    if (engine_file.good()) {
        std::cout << "[INFO] Loading pre-built engine: " << engine_path << std::endl;
        engine_file.seekg(0, std::ios::end);
        size_t size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        engine_file.read(buffer.data(), size);
        return TrtUniquePtr<nvinfer1::ICudaEngine>(runtime.deserializeCudaEngine(buffer.data(), size));
    }

    std::cout << "[INFO] Engine not found. Building from ONNX: " << model_path << std::endl;
    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
    TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));

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
        if (builder->platformHasFastFp16()) config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    TrtUniquePtr<nvinfer1::IHostMemory> serialized_engine(
            builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine) throw std::runtime_error("Failed to build engine.");

    std::ofstream p_engine(engine_path, std::ios::binary);
    p_engine.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());
    std::cout << "[INFO] Engine built and saved to: " << engine_path << std::endl;

    return TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime.deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));
}

void FaceAnalyzer::prepare(const std::string &provider,
                           float det_thresh,
                           cv::Size det_size) {
    std::string final_provider = (provider == "CUDAExecutionProvider" || provider == "GPU") ? "GPU" : "DLA";
    std::cout << "[INFO] Preparing FaceAnalyzer with backend: " << final_provider << std::endl;
    det_thresh_ = det_thresh;
    det_size_ = det_size;

    m_logger = std::make_unique<TrtLogger>();
    m_runtime.reset(nvinfer1::createInferRuntime(*m_logger));
    if (final_provider == "DLA") m_runtime->setDLACore(0);

    std::cout << "[INFO] Initializing detection model..." << std::endl;
    m_det_engine = loadOrCreateEngine(*m_runtime, m_det_model_path, final_provider, *m_logger);
    m_det_context.reset(m_det_engine->createExecutionContext());

    std::cout << "[INFO] Initializing recognition model..." << std::endl;
    m_rec_engine = loadOrCreateEngine(*m_runtime, m_rec_model_path, final_provider, *m_logger);
    m_rec_context.reset(m_rec_engine->createExecutionContext());

    cudaStreamCreate(&m_stream);

    auto sizeofTRT = [](nvinfer1::DataType type) -> size_t {
        switch (type) {
            case nvinfer1::DataType::kFLOAT:
                return 4;
            case nvinfer1::DataType::kHALF:
                return 2;
            case nvinfer1::DataType::kINT8:
                return 1;
            case nvinfer1::DataType::kINT32:
                return 4;
            default:
                throw std::runtime_error("Unknown TRT DataType size.");
        }
    };
    auto allocAllBindings = [&](TrtUniquePtr<nvinfer1::ICudaEngine> &engine,
                                TrtUniquePtr<nvinfer1::IExecutionContext> &ctx,
                                std::vector<void *> &buffers,
                                std::vector<size_t> &buf_sizes,
                                std::function<void(nvinfer1::IExecutionContext *, int)> setInputShape) {
        const int nb = engine->getNbBindings();
        buffers.assign(nb, nullptr);
        buf_sizes.assign(nb, 0);
        int inputIdx = findInputBinding(*engine);
        if (inputIdx < 0) throw std::runtime_error("Input binding not found.");
        setInputShape(ctx.get(), inputIdx);
        if (!ctx->allInputDimensionsSpecified())
            throw std::runtime_error("Input dimensions unspecified.");
        for (int i = 0; i < nb; ++i) {
            nvinfer1::Dims dims = ctx->getBindingDimensions(i);
            if (std::any_of(dims.d, dims.d + dims.nbDims, [](int v) { return v < 0; })) {
                if (engine->hasImplicitBatchDimension()) dims.d[0] = 1;
                else throw std::runtime_error("Binding has unresolved dimension.");
            }
            size_t vol = 1;
            for (int d = 0; d < dims.nbDims; ++d) vol *= static_cast<size_t>(dims.d[d]);
            size_t bytes = vol * sizeofTRT(engine->getBindingDataType(i));
            if (bytes == 0) continue;
            buf_sizes[i] = bytes;
            cudaMalloc(&buffers[i], bytes);
        }
    };

    allocAllBindings(m_det_engine, m_det_context, m_buffers_det, m_buffer_sizes_det,
                     [this](nvinfer1::IExecutionContext *c, int idx) {
                         c->setBindingDimensions(idx, nvinfer1::Dims4{1, 3, det_size_.height, det_size_.width});
                     });
    allocAllBindings(m_rec_engine, m_rec_context, m_buffers_rec, m_buffer_sizes_rec,
                     [](nvinfer1::IExecutionContext *c, int idx) {
                         c->setBindingDimensions(idx, nvinfer1::Dims4{1, 3, 112, 112});
                     });

    m_is_prepared = true;
    std::cout << "[INFO] FaceAnalyzer is ready." << std::endl;
}

std::vector<Face> FaceAnalyzer::get(const cv::cuda::GpuMat &img) {
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

    if (!faces.empty()) {
        std::cout << "[FaceAnalyzer::get] total " << total_ms << " ms (det "
                  << det_ms << " ms, rec_all " << rec_ms_total << " ms) for "
                  << faces.size() << " faces.\n";
    }
    return faces;
}

std::vector<Face> FaceAnalyzer::detect(const cv::cuda::GpuMat &img) {
    if (!m_is_prepared) throw std::runtime_error("FaceAnalyzer not prepared.");
    if (img.empty()) return {};

    const int orig_h = img.rows, orig_w = img.cols;
    float scale_ratio = static_cast<float>(det_size_.height) / std::max(orig_h, orig_w);
    int new_w = static_cast<int>(orig_w * scale_ratio), new_h = static_cast<int>(orig_h * scale_ratio);

    // GPU resize+pad
    cv::cuda::GpuMat resized_img;
    cv::cuda::resize(img, resized_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::cuda::GpuMat padded_img(det_size_, img.type(), cv::Scalar(0, 0, 0));
    resized_img.copyTo(padded_img(cv::Rect(0, 0, new_w, new_h)));
    // GPU float32 & sub mean
    cv::cuda::GpuMat processed_img;
    padded_img.convertTo(processed_img, CV_32FC3);
    cv::Scalar meanBGR(104, 117, 123);
    cv::cuda::subtract(processed_img, meanBGR, processed_img);

    // HWC->CHW pack to float host buffer (分通道download再memcpy)
    std::vector<float> input_data(det_size_.height * det_size_.width * 3);
    std::vector<cv::cuda::GpuMat> chs;
    cv::cuda::split(processed_img, chs);
    for (int c = 0; c < 3; ++c) {
        cv::Mat ch_host;
        chs[c].download(ch_host);
        std::memcpy(input_data.data() + c * det_size_.width * det_size_.height,
                    ch_host.ptr<float>(),
                    det_size_.width * det_size_.height * sizeof(float));
    }

    // inference
    int input_idx = findInputBinding(*m_det_engine, "input");
    if (input_idx < 0) throw std::runtime_error("Detection model must have an input binding named 'input'");
    int loc_idx = m_det_engine->getBindingIndex("boxes"),
            conf_idx = m_det_engine->getBindingIndex("scores"),
            landms_idx = m_det_engine->getBindingIndex("landmarks");
    if (loc_idx < 0 || conf_idx < 0 || landms_idx < 0)
        throw std::runtime_error("Detection model missing output binding.");

    cudaMemcpyAsync(m_buffers_det[input_idx], input_data.data(), m_buffer_sizes_det[input_idx], cudaMemcpyHostToDevice,
                    m_stream);
    m_det_context->enqueueV2(m_buffers_det.data(), m_stream, nullptr);

    auto loc_dims = m_det_context->getBindingDimensions(loc_idx);
    int num_priors = loc_dims.d[1];

    std::vector<float> loc_output(num_priors * 4),
            conf_output(num_priors * 2),
            landms_output(num_priors * 10);
    cudaMemcpyAsync(loc_output.data(), m_buffers_det[loc_idx], loc_output.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, m_stream);
    cudaMemcpyAsync(conf_output.data(), m_buffers_det[conf_idx], conf_output.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, m_stream);
    cudaMemcpyAsync(landms_output.data(), m_buffers_det[landms_idx], landms_output.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    const int top_k = 5000;
    const float nms_threshold = 0.4f;
    const int keep_top_k = 750;
    auto priors = generate_priors(det_size_.height, det_size_.width);
    auto boxes_normalized = decode_boxes(loc_output, priors);
    auto landms_normalized = decode_landmarks(landms_output, priors);

    std::vector<Detection> dets_raw;
    for (int i = 0; i < num_priors; ++i) {
        float score = conf_output[i * 2 + 1];
        if (score > det_thresh_) {
            Detection det;
            det.score = score;
            det.box.x1 = boxes_normalized[i].x1 * det_size_.width;
            det.box.y1 = boxes_normalized[i].y1 * det_size_.height;
            det.box.x2 = boxes_normalized[i].x2 * det_size_.width;
            det.box.y2 = boxes_normalized[i].y2 * det_size_.height;
            for (int j = 0; j < 5; ++j) {
                det.landmark.x_coords[j] = landms_normalized[i].x_coords[j] * det_size_.width;
                det.landmark.y_coords[j] = landms_normalized[i].y_coords[j] * det_size_.height;
            }
            dets_raw.push_back(det);
        }
    }
    if (dets_raw.empty()) return {};

    std::sort(dets_raw.begin(), dets_raw.end(), [](const Detection &a, const Detection &b) {
        return a.score > b.score;
    });
    if (dets_raw.size() > top_k) dets_raw.resize(top_k);

    std::vector<int> keep = cpu_nms(dets_raw, nms_threshold);
    std::vector<Face> faces;
    for (int idx: keep) {
        if (faces.size() >= keep_top_k) break;
        const auto &b = dets_raw[idx];
        Face f;
        f.det_score = b.score;
        f.bbox.x = b.box.x1 / scale_ratio;
        f.bbox.y = b.box.y1 / scale_ratio;
        f.bbox.width = (b.box.x2 - b.box.x1) / scale_ratio;
        f.bbox.height = (b.box.y2 - b.box.y1) / scale_ratio;
        for (int j = 0; j < 5; ++j) {
            float kps_x = b.landmark.x_coords[j] / scale_ratio;
            float kps_y = b.landmark.y_coords[j] / scale_ratio;
            f.kps.emplace_back(kps_x, kps_y);
        }
        faces.push_back(f);
    }
    return faces;
}

void FaceAnalyzer::get_embedding(const cv::cuda::GpuMat &full_img, Face &face) {
    if (face.kps.empty()) {
        face.embedding = cv::Mat();
        return;
    }
    if (face.kps.size() != 5) throw std::runtime_error("Face object must have 5 keypoints for alignment.");
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
    cv::cuda::GpuMat aligned;
    cv::cuda::warpAffine(full_img, aligned, M, cv::Size(112, 112));
    face.aligned_face.release();
    aligned.download(face.aligned_face);
    face.embedding = get_embedding_from_aligned(aligned);
}

cv::Mat FaceAnalyzer::get_embedding_from_aligned(const cv::cuda::GpuMat &aligned_img) {
    if (!m_is_prepared)
        throw std::runtime_error("FaceAnalyzer not prepared.");
    if (aligned_img.size() != cv::Size(112, 112))
        throw std::runtime_error("Input for get_embedding_from_aligned must be 112x112.");

    cv::cuda::GpuMat float_img;
    aligned_img.convertTo(float_img, CV_32FC3);
    cv::Scalar meanBGR(127.5, 127.5, 127.5);
    cv::cuda::subtract(float_img, meanBGR, float_img);
    cv::cuda::multiply(float_img, cv::Scalar(1.0 / 128.0), float_img);

    std::vector<float> preprocessed(1 * 3 * 112 * 112);
    std::vector<cv::cuda::GpuMat> chs;
    cv::cuda::split(float_img, chs);
    for (int c = 0; c < 3; ++c) {
        cv::Mat ch_host;
        chs[c].download(ch_host);
        std::memcpy(preprocessed.data() + c * 112 * 112, ch_host.ptr<float>(), 112 * 112 * sizeof(float));
    }

    int input_idx = findInputBinding(*m_rec_engine);
    int output_idx = findOutputBinding(*m_rec_engine);
    if (input_idx < 0 || output_idx < 0)
        throw std::runtime_error("Failed to find input/output binding in rec engine.");

    cudaMemcpyAsync(m_buffers_rec[input_idx], preprocessed.data(), m_buffer_sizes_rec[input_idx],
                    cudaMemcpyHostToDevice, m_stream);
    m_rec_context->enqueueV2(m_buffers_rec.data(), m_stream, nullptr);
    cv::Mat result(1, m_buffer_sizes_rec[output_idx] / sizeof(float), CV_32F);
    cudaMemcpyAsync(result.ptr<float>(), m_buffers_rec[output_idx], m_buffer_sizes_rec[output_idx],
                    cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);
    return result.clone();
}