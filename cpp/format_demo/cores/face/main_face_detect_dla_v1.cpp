#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <map>
#include <memory>
#include <iomanip> // Required for std::setprecision

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

// Helper to check for CUDA errors
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error in " << __FILE__ << " at line "     \
                      << __LINE__ << ": " << cudaGetErrorString(err)     \
                      << std::endl;                                      \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        // Suppress info-level messages
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

// Add a universal deleter for all TensorRT interface objects.
// This replaces the incorrect and complex decltype logic.
struct InferDeleter {
    template<typename T>
    void operator()(T *obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

// --- Configuration matching python cfg_mnet ---
struct ModelConfig {
    std::string name = "mobilenet0.25";
    std::vector<std::vector<int>> min_sizes = {{16,  32},
                                               {64,  128},
                                               {256, 512}};
    std::vector<int> steps = {8, 16, 32};
    std::vector<float> variance = {0.1f, 0.2f};
    bool clip = false;
};

// --- Structures for decoded results ---
struct BBox {
    float x1, y1, x2, y2;
};

struct Landmark {
    float x_coords[5];
    float y_coords[5];
};

struct Detection {
    BBox box;
    float score;
    Landmark landmark;
};

// --- Post-processing functions matching python implementations ---

// Generates prior boxes, exactly matching layers/functions/prior_box.py
std::vector<BBox> generate_priors(int im_height, int im_width) {
    ModelConfig cfg;
    std::vector<std::pair<int, int>> feature_maps;
    for (const auto &step: cfg.steps) {
        feature_maps.push_back({(int) ceil((float) im_height / step), (int) ceil((float) im_width / step)});
    }

    std::vector<BBox> anchors;
    for (size_t k = 0; k < feature_maps.size(); ++k) {
        auto f = feature_maps[k];
        auto min_sizes_k = cfg.min_sizes[k];
        for (int i = 0; i < f.first; ++i) {
            for (int j = 0; j < f.second; ++j) {
                for (const auto &min_size: min_sizes_k) {
                    float s_kx = (float) min_size / im_width;
                    float s_ky = (float) min_size / im_height;
                    float cx = (j + 0.5f) * cfg.steps[k] / im_width;
                    float cy = (i + 0.5f) * cfg.steps[k] / im_height;
                    anchors.push_back({cx, cy, s_kx, s_ky}); // Storing as cx, cy, w, h
                }
            }
        }
    }
    return anchors;
}

// Decodes boxes, exactly matching utils/box_utils.py's decode
std::vector<BBox> decode_boxes(const std::vector<float> &loc, const std::vector<BBox> &priors) {
    ModelConfig cfg;
    std::vector<BBox> boxes;
    boxes.reserve(priors.size());

    for (size_t i = 0; i < priors.size(); ++i) {
        float p_cx = priors[i].x1;
        float p_cy = priors[i].y1;
        float p_w = priors[i].x2;
        float p_h = priors[i].y2;

        float loc_x = loc[i * 4 + 0];
        float loc_y = loc[i * 4 + 1];
        float loc_w = loc[i * 4 + 2];
        float loc_h = loc[i * 4 + 3];

        float cx = p_cx + loc_x * cfg.variance[0] * p_w;
        float cy = p_cy + loc_y * cfg.variance[0] * p_h;
        float w = p_w * exp(loc_w * cfg.variance[1]);
        float h = p_h * exp(loc_h * cfg.variance[1]);

        boxes.push_back({
                                cx - w / 2,
                                cy - h / 2,
                                cx + w / 2,
                                cy + h / 2
                        });
    }
    return boxes;
}

// Decodes landmarks, exactly matching utils/box_utils.py's decode_landm
std::vector<Landmark> decode_landmarks(const std::vector<float> &pre, const std::vector<BBox> &priors) {
    ModelConfig cfg;
    std::vector<Landmark> landmarks;
    landmarks.reserve(priors.size());

    for (size_t i = 0; i < priors.size(); ++i) {
        float p_cx = priors[i].x1;
        float p_cy = priors[i].y1;
        float p_w = priors[i].x2;
        float p_h = priors[i].y2;

        Landmark l;
        for (int j = 0; j < 5; ++j) {
            l.x_coords[j] = p_cx + pre[i * 10 + j * 2] * cfg.variance[0] * p_w;
            l.y_coords[j] = p_cy + pre[i * 10 + j * 2 + 1] * cfg.variance[0] * p_h;
        }
        landmarks.push_back(l);
    }
    return landmarks;
}

// NMS, exactly matching utils/nms/py_cpu_nms.py
std::vector<int> cpu_nms(std::vector<Detection> &dets, float thresh) {
    std::vector<int> keep;
    if (dets.empty()) {
        return keep;
    }

    std::vector<size_t> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t i, size_t j) {
        return dets[i].score > dets[j].score;
    });

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

            float xx1 = std::max(box_i.x1, box_j.x1);
            float yy1 = std::max(box_i.y1, box_j.y1);
            float xx2 = std::min(box_i.x2, box_j.x2);
            float yy2 = std::min(box_i.y2, box_j.y2);

            float w = std::max(0.0f, xx2 - xx1 + 1);
            float h = std::max(0.0f, yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[current_idx] - inter);

            if (ovr <= thresh) {
                new_order.push_back(current_idx);
            }
        }
        order = new_order;
    }

    return keep;
}

int main(int argc, char **argv) {
    // --- Arguments (hardcoded for simplicity, matching python script args) ---
    const std::string onnx_path = "/mnt/nfs/mobilenet0.25_Final.onnx";
    const std::string engine_path = "/mnt/nfs/mobilenet0.25_Final.dla.engine";
    const float confidence_threshold = 0.02f;
    const int top_k = 5000;
    const float nms_threshold = 0.4f;
    const int keep_top_k = 750;
    const float vis_thres = 0.6f;
    const std::string image_path = "/mnt/nfs/padded_test.bmp";
    const std::string save_img_path = "/mnt/nfs/test_cpp.jpg";

    // Create directory for saving if it doesn't exist
    std::string save_dir = save_img_path.substr(0, save_img_path.find_last_of("/"));
    system(("mkdir -p " + save_dir).c_str());

    // --- Initialize ---
    Logger gLogger;
    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine;

    std::ifstream engine_file(engine_path, std::ios::binary);
    if (engine_file.good()) {
        std::cout << "Loading engine from " << engine_path << std::endl;
        engine_file.seekg(0, engine_file.end);
        size_t size = engine_file.tellg();
        engine_file.seekg(0, engine_file.beg);

        std::vector<char> engine_data(size);
        engine_file.read(engine_data.data(), size);

        runtime.reset(nvinfer1::createInferRuntime(gLogger));
        engine.reset(runtime->deserializeCudaEngine(engine_data.data(), size));

    } else {
        std::cout << "Building engine from " << onnx_path << "..." << std::endl;
        auto builder = std::unique_ptr<nvinfer1::IBuilder, InferDeleter>(nvinfer1::createInferBuilder(gLogger));
        const auto explicitBatch =
                1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>(
                builder->createNetworkV2(explicitBatch));
        auto parser = std::unique_ptr<nvonnxparser::IParser, InferDeleter>(
                nvonnxparser::createParser(*network, gLogger));

        if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "Failed to parse ONNX file." << std::endl;
            return -1;
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter>(builder->createBuilderConfig());
        config->setMaxWorkspaceSize(1LL << 30); // 1GB

        // --- DLA Configuration ---
        std::cout << "Configuring for DLA." << std::endl;
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(0); // Use DLA core 0. Change to 1 for the other core.
        config->setFlag(nvinfer1::BuilderFlag::kFP16); // DLA usually prefers FP16 or INT8
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK); // Allow layers unsupported on DLA to run on GPU

        // Get input tensor and set dynamic shapes if needed
        auto input = network->getInput(0);
        auto input_dims = input->getDimensions();
        // Here we assume a fixed size matching python logic which doesn't resize
        // If your ONNX was exported with dynamic shapes, you can set optimization profiles here.

        // Build and serialize engine
        std::unique_ptr<nvinfer1::IHostMemory, InferDeleter> serialized_engine(
                builder->buildSerializedNetwork(*network, *config));
        if (!serialized_engine) {
            std::cerr << "Failed to build engine." << std::endl;
            return -1;
        }

        std::ofstream out_engine_file(engine_path, std::ios::binary);
        out_engine_file.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

        // Reload from serialized engine
        runtime.reset(nvinfer1::createInferRuntime(gLogger));
        engine.reset(runtime->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));
    }

    if (!engine) {
        std::cerr << "Engine creation failed." << std::endl;
        return -1;
    }

    auto context = std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter>(engine->createExecutionContext());

    // --- Pre-processing ---
    cv::Mat img_raw = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img_raw.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    cv::Mat img;
    img_raw.convertTo(img, CV_32FC3);

    const int im_height = img.rows;
    const int im_width = img.cols;

    // Subtract mean: (104, 117, 123) - BGR order from OpenCV
    cv::subtract(img, cv::Scalar(104, 117, 123), img);

    // HWC to CHW and copy to a flat vector
    std::vector<float> input_data(im_height * im_width * 3);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < im_height; ++h) {
            for (int w = 0; w < im_width; ++w) {
                input_data[c * im_width * im_height + h * im_width + w] = img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    // --- Inference ---
    void *buffers[4]; // 1 input, 3 outputs
    const int input_idx = engine->getBindingIndex("input");

    // Set input tensor dimensions
    context->setBindingDimensions(input_idx, nvinfer1::Dims4{1, 3, im_height, im_width});

    if (!context->allInputDimensionsSpecified()) {
        std::cerr << "Not all input dimensions are specified." << std::endl;
        return -1;
    }

    CUDA_CHECK(cudaMalloc(&buffers[input_idx], 1 * 3 * im_height * im_width * sizeof(float)));

    const int loc_idx = engine->getBindingIndex("boxes");
    const int conf_idx = engine->getBindingIndex("scores");
    const int landms_idx = engine->getBindingIndex("landmarks");

    auto loc_dims = context->getBindingDimensions(loc_idx);
    auto conf_dims = context->getBindingDimensions(conf_idx);
    auto landms_dims = context->getBindingDimensions(landms_idx);

    int num_priors = loc_dims.d[1];
    std::cout << "Number of priors detected: " << num_priors << std::endl;

    std::vector<float> loc_output(num_priors * 4);
    std::vector<float> conf_output(num_priors * 2);
    std::vector<float> landms_output(num_priors * 10);

    CUDA_CHECK(cudaMalloc(&buffers[loc_idx], num_priors * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[conf_idx], num_priors * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[landms_idx], num_priors * 10 * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(buffers[input_idx], input_data.data(), input_data.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    auto start = std::chrono::high_resolution_clock::now();
    context->enqueueV2(buffers, stream, nullptr);
    auto end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpyAsync(loc_output.data(), buffers[loc_idx], loc_output.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(conf_output.data(), buffers[conf_idx], conf_output.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(landms_output.data(), buffers[landms_idx], landms_output.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
    std::chrono::duration<double, std::milli> net_forward_time = end - start;
    std::cout << "net forward time: " << net_forward_time.count() / 1000.0 << " s" << std::endl;

    // --- Post-processing (exactly matching python script) ---
    auto priors_center_form = generate_priors(im_height, im_width);

    // The python code calculates softmax on the confidence scores.
    // Our ONNX model already has softmax. But in case it doesn't:
    // for (int i = 0; i < num_priors; ++i) {
    //     float e0 = exp(conf_output[i * 2 + 0]);
    //     float e1 = exp(conf_output[i * 2 + 1]);
    //     conf_output[i * 2 + 0] = e0 / (e0 + e1);
    //     conf_output[i * 2 + 1] = e1 / (e0 + e1);
    // }

    auto boxes_normalized = decode_boxes(loc_output, priors_center_form);
    auto landms_normalized = decode_landmarks(landms_output, priors_center_form);

    std::vector<Detection> dets_raw;
    for (int i = 0; i < num_priors; ++i) {
        float score = conf_output[i * 2 + 1];
        if (score > confidence_threshold) {
            Detection det;
            det.score = score;
            det.box.x1 = boxes_normalized[i].x1 * im_width;
            det.box.y1 = boxes_normalized[i].y1 * im_height;
            det.box.x2 = boxes_normalized[i].x2 * im_width;
            det.box.y2 = boxes_normalized[i].y2 * im_height;

            for (int j = 0; j < 5; ++j) {
                det.landmark.x_coords[j] = landms_normalized[i].x_coords[j] * im_width;
                det.landmark.y_coords[j] = landms_normalized[i].y_coords[j] * im_height;
            }
            dets_raw.push_back(det);
        }
    }

    // Keep top-K before NMS
    std::sort(dets_raw.begin(), dets_raw.end(), [](const Detection &a, const Detection &b) {
        return a.score > b.score;
    });
    if (dets_raw.size() > top_k) {
        dets_raw.resize(top_k);
    }

    // Do NMS
    std::vector<int> keep = cpu_nms(dets_raw, nms_threshold);

    std::vector<Detection> final_dets;
    for (int idx: keep) {
        final_dets.push_back(dets_raw[idx]);
    }

    // Keep top-K after NMS
    if (final_dets.size() > keep_top_k) {
        final_dets.resize(keep_top_k);
    }

    // --- Save results ---
    std::string save_txt_path = save_img_path.substr(0, save_img_path.find_last_of(".")) + ".txt";
    std::ofstream f_txt(save_txt_path);

    for (const auto &b: final_dets) {
        if (b.score < vis_thres) continue;

        // Write to txt file
        f_txt << static_cast<int>(b.box.x1) << " " << static_cast<int>(b.box.y1) << " "
              << static_cast<int>(b.box.x2) << " " << static_cast<int>(b.box.y2) << " "
              << std::fixed << std::setprecision(5) << b.score;
        for (int j = 0; j < 5; ++j) {
            f_txt << " " << static_cast<int>(b.landmark.x_coords[j]) << " " << static_cast<int>(b.landmark.y_coords[j]);
        }
        f_txt << "\n";

        // Draw on image
        cv::rectangle(img_raw, cv::Point((int) b.box.x1, (int) b.box.y1), cv::Point((int) b.box.x2, (int) b.box.y2),
                      cv::Scalar(0, 0, 255), 2);

        std::stringstream score_stream;
        score_stream << std::fixed << std::setprecision(4) << b.score;
        std::string text = score_stream.str();
        cv::putText(img_raw, text, cv::Point((int) b.box.x1, (int) b.box.y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5,
                    cv::Scalar(255, 255, 255));

        // Landmarks
        cv::circle(img_raw, cv::Point((int) b.landmark.x_coords[0], (int) b.landmark.y_coords[0]), 1,
                   cv::Scalar(0, 0, 255), 4);
        cv::circle(img_raw, cv::Point((int) b.landmark.x_coords[1], (int) b.landmark.y_coords[1]), 1,
                   cv::Scalar(0, 255, 255), 4);
        cv::circle(img_raw, cv::Point((int) b.landmark.x_coords[2], (int) b.landmark.y_coords[2]), 1,
                   cv::Scalar(255, 0, 255), 4);
        cv::circle(img_raw, cv::Point((int) b.landmark.x_coords[3], (int) b.landmark.y_coords[3]), 1,
                   cv::Scalar(0, 255, 0), 4);
        cv::circle(img_raw, cv::Point((int) b.landmark.x_coords[4], (int) b.landmark.y_coords[4]), 1,
                   cv::Scalar(255, 0, 0), 4);
    }

    f_txt.close();
    cv::imwrite(save_img_path, img_raw);
    std::cout << "Detection results saved to " << save_img_path << " and " << save_txt_path << std::endl;

    // --- Cleanup ---
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[input_idx]));
    CUDA_CHECK(cudaFree(buffers[loc_idx]));
    CUDA_CHECK(cudaFree(buffers[conf_idx]));
    CUDA_CHECK(cudaFree(buffers[landms_idx]));

    return 0;
}