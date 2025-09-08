#ifndef FACEEMBEDDINGEXTRACTOR_FACEANALYZER_HPP
#define FACEEMBEDDINGEXTRACTOR_FACEANALYZER_HPP

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

#include <NvInfer.h>

// 定义TRT对象的自定义Deleter (非匿名)
template<typename T>
struct TrtDeleter {
    void operator()(T *obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

// 为带有自定义Deleter的unique_ptr创建别名
template<typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

struct Face {
    cv::Rect2d bbox;
    float det_score = 0.0f;
    std::vector<cv::Point2f> kps;
    cv::Mat embedding;
    cv::Mat aligned_face;

    Face() = default;
};

class FaceAnalyzer {
public:
    FaceAnalyzer(const std::string &det_model_path, const std::string &rec_model_path);

    ~FaceAnalyzer();

    void prepare(const std::string &provider, float det_thresh, cv::Size det_size);

    std::vector<Face> get(const cv::Mat &img);

    // MODIFIED HERE: Moved 'detect' from private to public
    std::vector<Face> detect(const cv::Mat &img);

    // Recognition-related methods are now public as per full functionality
    void get_embedding(const cv::Mat &full_img, Face &face);

    cv::Mat get_embedding_from_aligned(const cv::Mat &aligned_img);

private:
    // TensorRT 核心组件
    std::unique_ptr<nvinfer1::ILogger> m_logger;
    TrtUniquePtr<nvinfer1::IRuntime> m_runtime;

    TrtUniquePtr<nvinfer1::ICudaEngine> m_det_engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> m_det_context;
    std::vector<void *> m_buffers_det;
    std::vector<size_t> m_buffer_sizes_det;
    // Restored full list of output names
    std::vector<std::string> m_det_output_names = {"448", "471", "494", "451", "474", "497", "454", "477", "500"};

    TrtUniquePtr<nvinfer1::ICudaEngine> m_rec_engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> m_rec_context;
    std::vector<void *> m_buffers_rec;
    std::vector<size_t> m_buffer_sizes_rec;

    // CUDA 资源
    cudaStream_t m_stream;

    // 配置和路径
    std::string m_det_model_path;
    std::string m_rec_model_path;
    float det_thresh_ = 0.5f;
    cv::Size det_size_ = {640, 640};
    bool m_is_prepared = false;
};

#endif //FACEEMBEDDINGEXTRACTOR_FACEANALYZER_HPP