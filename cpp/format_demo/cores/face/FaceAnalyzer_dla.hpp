#ifndef FACEEMBEDDINGEXTRACTOR_FACEANALYZER_HPP
#define FACEEMBEDDINGEXTRACTOR_FACEANALYZER_HPP

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

template<typename T>
struct TrtDeleter {
    void operator()(T *obj) const {
        if (obj) { obj->destroy(); }
    }
};

template<typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

// 辅助结构体
struct BBox {
    float x1, y1, x2, y2;
};
struct Landmark {
    float x_coords[5];
    float y_coords[5];
};
struct FaceDet {
    BBox box;
    float score;
    Landmark landmark;
};

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

    // 全部接口都收/传cv::cuda::GpuMat
    std::vector<Face> get(const cv::cuda::GpuMat &img);

    std::vector<Face> detect(const cv::cuda::GpuMat &img);

    void get_embedding(const cv::cuda::GpuMat &full_img, Face &face);

    cv::Mat get_embedding_from_aligned(const cv::cuda::GpuMat &aligned_img);

private:
    std::unique_ptr<nvinfer1::ILogger> m_logger;
    TrtUniquePtr<nvinfer1::IRuntime> m_runtime;

    TrtUniquePtr<nvinfer1::ICudaEngine> m_det_engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> m_det_context;
    std::vector<void *> m_buffers_det;
    std::vector<size_t> m_buffer_sizes_det;
    std::vector<std::string> m_det_output_names = {"boxes", "scores", "landmarks"};

    TrtUniquePtr<nvinfer1::ICudaEngine> m_rec_engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> m_rec_context;
    std::vector<void *> m_buffers_rec;
    std::vector<size_t> m_buffer_sizes_rec;

    cudaStream_t m_stream;

    std::string m_det_model_path;
    std::string m_rec_model_path;
    float det_thresh_ = 0.5f;
    cv::Size det_size_ = {640, 640};
    bool m_is_prepared = false;
};

#endif //FACEEMBEDDINGEXTRACTOR_FACEANALYZER_HPP