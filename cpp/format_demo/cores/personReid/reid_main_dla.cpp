#include "PersonReid_dla.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

namespace fs = std::filesystem;

int main() {
    // ########################### 用户配置 ###########################
    std::string resized_input_dir = "/mnt/nfs/out_reid_resized";
    std::string onnx_model_path = "/mnt/nfs/reid_model.onnx";
    std::string engine_cache_path = "/mnt/nfs/reid_model_dla.engine"; // DLA引擎缓存文件路径
    int dla_core_id = 0; // 使用 DLA 核心 0 (Jetson AGX/Orin 系列可选 0 或 1)

    const int MODEL_INPUT_WIDTH = 128;
    const int MODEL_INPUT_HEIGHT = 256;
    // ##################################################################

    std::cout << "================ 任务1: 加载图像路径 ================" << std::endl;
    std::vector<fs::path> bmp_paths;
    for (const auto &entry: fs::directory_iterator(resized_input_dir)) {
        if (entry.path().extension() == ".bmp") {
            bmp_paths.push_back(entry.path());
        }
    }

    if (bmp_paths.empty()) {
        std::cerr << "错误: 在 \"" << resized_input_dir << "\" 中未找到 BMP 图像!" << std::endl;
        return -1;
    }

    // 按图片文件名中的数字排序
    std::sort(bmp_paths.begin(), bmp_paths.end(), [](const fs::path &a, const fs::path &b) {
        return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
    });
    std::cout << "找到 " << bmp_paths.size() << " 张待处理的图片。" << std::endl;

    std::cout << "\n================ 任务2: DLA 推理 (C++) ================" << std::endl;
    try {
        std::cout << "初始化 ReID DLA 模型..." << std::endl;
        PersonReidDLA reid(onnx_model_path, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, dla_core_id, engine_cache_path);
        std::cout << "DLA 模型加载完成。" << std::endl;

        std::cout << "开始从 " << bmp_paths.size() << " 张图片中提取特征..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < bmp_paths.size(); ++i) {
            cv::Mat img = cv::imread(bmp_paths[i].string(), cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "警告: 无法读取图片 " << bmp_paths[i] << std::endl;
                continue;
            }
            cv::Mat feat = reid.extract_feat(img);

            if ((i + 1) % 100 == 0 || (i + 1) == bmp_paths.size()) {
                std::cout << "  [" << (i + 1) << "/" << bmp_paths.size() << "] 已提取。 "
                          << "示例特征L2范数: " << cv::norm(feat) << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "特征提取完成，耗时 " << std::fixed << std::setprecision(2) << elapsed.count() << " 秒。"
                  << std::endl;
        std::cout << "平均每张图片耗时 " << std::fixed << std::setprecision(2)
                  << (elapsed.count() * 1000 / bmp_paths.size()) << " 毫秒。" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "发生严重错误: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "\n==============================================" << std::endl;
    std::cout << "DLA 推理任务全部完成!" << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}