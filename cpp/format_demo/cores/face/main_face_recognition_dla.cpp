#include "FaceAnalyzer_dla.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

int main() {
    std::cout << "\n\n[INFO] ===== NEW MAIN using FaceAnalyzer::get_embedding_from_aligned =====\n\n";

    // ---------------- 配置 ----------------
    std::string bmp_folder_path = "/mnt/nfs/face_aligned_py_bmp/";      // 已裁剪对齐到112x112的bmp
    std::string output_txt_path = "/mnt/nfs/embeddings_cpp_from_aligned_bmps.txt";
    std::string rec_model_path = "/mnt/nfs/w600k_r50_simplified.onnx";
    // 假设没有用到 det_model，这里传一个空即可
    std::string det_model_path = "/mnt/nfs/mobilenet0.25_Final.onnx";

    try {
        // 1. 初始化 FaceAnalyzer，只准备识别模型
        FaceAnalyzer analyzer(det_model_path, rec_model_path);
        analyzer.prepare("DLA", 0.5f, cv::Size(640, 640));  // prepare必须执行，但我们只用rec模型

        // 2. 打开输出文件
        std::ofstream out_file(output_txt_path);
        if (!out_file.is_open()) {
            throw std::runtime_error("Failed to open output file: " + output_txt_path);
        }

        // 3. 收集 & 排序目录下所有 BMP
        std::vector<fs::path> bmp_paths;
        if (!fs::exists(bmp_folder_path) || !fs::is_directory(bmp_folder_path)) {
            throw std::runtime_error("Input folder does not exist: " + bmp_folder_path);
        }
        for (const auto &entry: fs::directory_iterator(bmp_folder_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bmp") {
                bmp_paths.push_back(entry.path());
            }
        }
        std::sort(bmp_paths.begin(), bmp_paths.end());
        std::cout << "[INFO] Found " << bmp_paths.size() << " BMP files.\n";

        // 4. 遍历图像并提取embedding
        for (const auto &bmp_path: bmp_paths) {
            cv::Mat img = cv::imread(bmp_path.string());
            if (img.empty() || img.size() != cv::Size(112, 112)) {
                std::cerr << "[WARNING] Invalid image (skip): " << bmp_path << "\n";
                continue;
            }

            // 上传到GPU（接口签名要求GpuMat）
            cv::cuda::GpuMat gpu_img;
            gpu_img.upload(img);

            // 提取embedding向量
            cv::Mat embedding = analyzer.get_embedding_from_aligned(gpu_img);

            if (embedding.empty()) {
                std::cerr << "[WARNING] Empty embedding: " << bmp_path << "\n";
                continue;
            }

            // L2归一化
            cv::Mat embedding_norm;
            cv::normalize(embedding, embedding_norm, 1.0, 0.0, cv::NORM_L2);

            // === 写入文件 ===
            std::string face_id = bmp_path.stem().string();
            out_file << face_id;
            const float *data = embedding_norm.ptr<float>(0);
            for (int i = 0; i < embedding_norm.cols; ++i) {
                out_file << "," << std::fixed << std::setprecision(8) << data[i];
            }
            out_file << "\n";

            std::cout << "[INFO] Processed " << face_id << "\n";
        }

        out_file.close();
        std::cout << "[SUCCESS] All embeddings saved to " << output_txt_path << "\n";

    } catch (const std::exception &e) {
        std::cerr << "[ERROR] exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}