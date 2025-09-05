#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <chrono>

#include <opencv2/opencv.hpp>
#include "PersonReid.hpp"

namespace fs = std::filesystem;

void save_feats_to_txt(const std::vector<cv::Mat> &feats, const std::vector<std::string> &img_names,
                       const std::string &output_path) {
    std::cout << "\nSaving features to \"" << output_path << "\"..." << std::endl;
    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << output_path << std::endl;
        return;
    }

    outfile << std::fixed << std::setprecision(8);

    for (size_t i = 0; i < feats.size(); ++i) {
        std::string basename = fs::path(img_names[i]).stem().string();
        outfile << basename;

        const cv::Mat &feat_vec = feats[i];
        for (int j = 0; j < feat_vec.cols; ++j) {
            outfile << " " << feat_vec.at<float>(0, j);
        }
        outfile << "\n";
    }

    outfile.close();
    std::cout << "Features saved successfully." << std::endl;
}

int main() {
    // ########################### USER CONFIGURATION ###########################
    std::string resized_input_dir = "/mnt/nfs/out_reid_resized"; // Python 已经 resize 好的目录
    std::string onnx_model_path = "/mnt/nfs/reid_model.onnx";
    std::string output_txt_path_cpp = "/mnt/nfs/features_cpp_onnx_arm.txt";

    const int MODEL_INPUT_HEIGHT = 256;
    const int MODEL_INPUT_WIDTH = 128;
    const bool USE_GPU = true;
    // ##########################################################################

    // ------------------- Task 1: Load resized images -------------------
    std::cout << "================ Task 1: Loading Resized Images ================" << std::endl;

    std::vector<fs::path> bmp_paths;
    for (const auto &entry: fs::directory_iterator(resized_input_dir))
        if (entry.path().extension() == ".bmp")
            bmp_paths.push_back(entry.path());

    std::sort(bmp_paths.begin(), bmp_paths.end(), [](const fs::path &a, const fs::path &b) {
        return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
    });

    std::vector<std::string> bmp_path_strings;
    for (const auto &p: bmp_paths) bmp_path_strings.push_back(p.string());

    if (bmp_paths.empty()) {
        std::cerr << "Fatal: No BMP images found in \"" << resized_input_dir << "\"!" << std::endl;
        return -1;
    }

    // ------------------- Task 2: Extract features -------------------
    std::cout << "================ Task 2: ONNX Inference (C++) ================" << std::endl;

    try {
        std::cout << "Initializing ReID model with OpenCV DNN..." << std::endl;
        PersonReid reid(onnx_model_path, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, USE_GPU);
        std::cout << "Model loaded." << std::endl;

        std::cout << "Starting feature extraction from " << bmp_paths.size() << " BMP images..." << std::endl;
        std::vector<cv::Mat> all_feats;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < bmp_paths.size(); ++i) {
            cv::Mat img = cv::imread(bmp_paths[i].string(), cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Warning: Could not read image " << bmp_paths[i] << std::endl;
                continue;
            }
            all_feats.push_back(reid.extract_feat(img));
            if ((i + 1) % 200 == 0 || (i + 1) == bmp_paths.size()) {
                std::cout << "  [" << (i + 1) << "/" << bmp_paths.size() << "] extracted" << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "Feature extraction finished in " << std::fixed << std::setprecision(1) << elapsed.count() << "s"
                  << std::endl;

        save_feats_to_txt(all_feats, bmp_path_strings, output_txt_path_cpp);
    } catch (const std::exception &e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return -1;
    }

    // ------------------- Summary -------------------
    std::cout << "\n==============================================" << std::endl;
    std::cout << "Task complete!" << std::endl;
    std::cout << "  - Input resized BMP images: " << fs::absolute(resized_input_dir).string() << std::endl;
    std::cout << "  - C++ ONNX features:        " << fs::absolute(output_txt_path_cpp).string() << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}