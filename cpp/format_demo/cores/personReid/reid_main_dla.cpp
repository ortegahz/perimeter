#include "PersonReid_dla.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>

namespace fs = std::filesystem;

/**
 * @brief 将 GPU 特征向量保存到文本文件
 * @param feats 特征向量的 vector (GpuMat，每个都是 1xN)
 * @param img_names 对应的原始图片路径
 * @param output_path 输出的 txt 文件路径
 */
void save_feats_to_txt(const std::vector<cv::cuda::GpuMat> &feats,
                       const std::vector<std::string> &img_names,
                       const std::string &output_path) {
    std::cout << "\n正在保存特征到 \"" << output_path << "\"..." << std::endl;
    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "错误: 无法打开文件进行写入: " << output_path << std::endl;
        return;
    }

    outfile << std::fixed << std::setprecision(8);

    for (size_t i = 0; i < feats.size(); ++i) {
        if (feats[i].empty()) continue;

        // GPU -> CPU 转换
        cv::Mat feat_cpu;
        feats[i].download(feat_cpu);

        // 提取不带扩展名的文件名
        std::string basename = fs::path(img_names[i]).stem().string();
        outfile << basename;

        for (int j = 0; j < feat_cpu.cols; ++j) {
            outfile << " " << feat_cpu.at<float>(0, j);
        }
        outfile << "\n";
    }

    outfile.close();
    std::cout << "特征保存成功。" << std::endl;
}

int main() {
    // ########################### 用户配置 ###########################
    std::string resized_input_dir = "/mnt/nfs/out_reid_resized";
    std::string onnx_model_path = "/mnt/nfs/reid_model.onnx";
    std::string engine_cache_path = "/mnt/nfs/reid_model_dla.engine";
    std::string output_txt_path_cpp = "/mnt/nfs/features_cpp_onnx_arm.txt"; // 输出路径
    int dla_core_id = 1;

    const int MODEL_INPUT_WIDTH = 128;
    const int MODEL_INPUT_HEIGHT = 256;
    // ##################################################################

    // ------------------- 任务1: 加载图像路径 -------------------
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

    std::sort(bmp_paths.begin(), bmp_paths.end(), [](const fs::path &a, const fs::path &b) {
        try {
            return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
        } catch (const std::invalid_argument &) {
            return a.stem().string() < b.stem().string();
        }
    });

    std::vector<std::string> bmp_path_strings;
    for (const auto &p: bmp_paths) {
        bmp_path_strings.push_back(p.string());
    }
    std::cout << "找到并排序了 " << bmp_paths.size() << " 张待处理的图片。" << std::endl;

    // ------------------- 任务2: DLA 推理并保存结果 -------------------
    std::cout << "\n================ 任务2: DLA 推理 (C++) ================" << std::endl;
    try {
        std::cout << "初始化 ReID DLA 模型..." << std::endl;
        PersonReidDLA reid(onnx_model_path, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, dla_core_id, engine_cache_path);
        std::cout << "DLA 模型加载完成。" << std::endl;

        const int runs_per_image = 1;

        std::cout << "开始从 " << bmp_paths.size() << " 张图片中提取特征 (每张图片运行 " << runs_per_image << " 次)..."
                  << std::endl;

        std::vector<cv::cuda::GpuMat> all_feats;
        all_feats.reserve(bmp_paths.size());
        auto total_start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < bmp_paths.size(); ++i) {
            cv::Mat img_cpu = cv::imread(bmp_path_strings[i], cv::IMREAD_COLOR);
            if (img_cpu.empty()) {
                std::cerr << "警告: 无法读取图片 " << bmp_path_strings[i] << std::endl;
                continue;
            }

            // 上传到 GPU
            cv::cuda::GpuMat img;
            img.upload(img_cpu);

            auto image_start_time = std::chrono::high_resolution_clock::now();

            cv::cuda::GpuMat last_feat;
            for (int j = 0; j < runs_per_image; ++j) {
                last_feat = reid.extract_feat(img);
            }
            all_feats.push_back(last_feat);

            auto image_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> image_elapsed = image_end_time - image_start_time;
            double avg_time_ms = (image_elapsed.count() * 1000.0) / runs_per_image;

            std::cout << "  - 处理 [" << i + 1 << "/" << bmp_paths.size() << "]: "
                      << fs::path(bmp_path_strings[i]).filename().string()
                      << "，平均耗时: " << std::fixed << std::setprecision(4)
                      << avg_time_ms << " 毫秒/次" << std::endl;
        }

        auto total_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_elapsed = total_end_time - total_start_time;

        long long total_inferences = bmp_paths.size() * runs_per_image;

        std::cout << "\n-------------------- 总体性能总结 --------------------" << std::endl;
        std::cout << "所有图片处理完成 (" << total_inferences << " 次总推理)，总耗时 "
                  << std::fixed << std::setprecision(2) << total_elapsed.count() << " 秒。" << std::endl;
        std::cout << "总体平均每次推理耗时 "
                  << std::fixed << std::setprecision(4)
                  << (total_elapsed.count() * 1000.0 / total_inferences)
                  << " 毫秒。" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;

        save_feats_to_txt(all_feats, bmp_path_strings, output_txt_path_cpp);

    } catch (const std::exception &e) {
        std::cerr << "\n发生严重错误: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "\n==============================================" << std::endl;
    std::cout << "任务全部完成!" << std::endl;
    std::cout << "  - 输入图片目录: " << fs::absolute(resized_input_dir).string() << std::endl;
    std::cout << "  - C++ DLA 特征输出: " << fs::absolute(output_txt_path_cpp).string() << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}