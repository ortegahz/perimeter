#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <filesystem>
#include <algorithm> // 需要此头文件用于排序

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// 使用 `std::filesystem` 需要此别名
namespace fs = std::filesystem;

int main() {
    // ======================= Configuration =======================
    // 1. 设置包含BMP文件的文件夹路径
    std::string bmp_folder_path = "/home/manu/tmp/face_aligned_py_bmp/";

    // 2. 设置统一的输出TXT文件的路径
    std::string output_txt_path = "/home/manu/tmp/embeddings_cpp_from_aligned_bmps.txt";

    // 3. 设置识别（recognition）模型的路径
    std::string rec_model_path = "/home/manu/.insightface/models/buffalo_l/w600k_r50_simplified.onnx";
    // =============================================================

    std::cout << "[INFO] Starting batch feature extraction test..." << std::endl;
    std::cout << "[INFO] Reading BMP images from: " << bmp_folder_path << std::endl;

    try {
        // --- 1. 加载ONNX识别模型 (只需加载一次) ---
        cv::dnn::Net rec_net = cv::dnn::readNetFromONNX(rec_model_path);
        if (rec_net.empty()) {
            throw std::runtime_error("Failed to load recognition model: " + rec_model_path);
        }

        // --- 2. 设置模型运行的后端和目标设备 (只需设置一次) ---
        rec_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        rec_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "[INFO] Recognition model loaded and set to CPU backend." << std::endl;

        // --- 3. 准备输出文件 (打开一次，准备写入) ---
        std::ofstream out_file(output_txt_path);
        if (!out_file.is_open()) {
            throw std::runtime_error("Failed to open output file for writing: " + output_txt_path);
        }

        // ======================= 【修改的部分在此】 =======================
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
            if (aligned_img.empty()) {
                std::cerr << "[WARNING] Failed to read image: " << bmp_path << std::endl;
                continue; // 跳过这个文件，继续处理下一个
            }

            // --- 创建输入Blob (swapRB=true) ---
            cv::Mat blob = cv::dnn::blobFromImage(aligned_img, 1.0 / 128.0, cv::Size(112, 112),
                                                  cv::Scalar(127.5, 127.5, 127.5), true, true);

            // --- 执行前向传播 ---
            rec_net.setInput(blob);
            cv::Mat embedding_raw = rec_net.forward();
            if (embedding_raw.empty()) {
                std::cerr << "[WARNING] Failed to get embedding for: " << bmp_path << std::endl;
                continue;
            }

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

        // --- 6. 关闭文件 ---
        out_file.close();
        // ======================= 【修改结束】 =======================

        std::cout << "[SUCCESS] All embeddings saved to " << output_txt_path << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "[ERROR] An exception occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}