/**
 * @file test_pose_pnp.cpp
 * @brief C++ version of the Python script to test ArcFace 5-point landmarks
 *        and calculate head pose (Yaw/Pitch/Roll) using solvePnP.
 *
 * Functionality:
 * 1. Recursively reads all images from a specified folder.
 * 2. Uses the existing `FaceAnalyzer` to detect faces and their 5 keypoints.
 * 3. Calculates 3D pose angles (Yaw/Pitch/Roll) via solvePnP.
 * 4. Prints results to the console and saves them to a TXT file in a
 *    format identical to the Python script for comparison.
 * 5. Optionally displays images with rendered bounding boxes, keypoints, and pose angles.
 */

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <optional>
#include <algorithm>

#include <opencv2/opencv.hpp>

// Include the project's existing FaceAnalyzer header
#include "cores/face/FaceAnalyzer.hpp"
// Include the new PoseEstimator header for pose calculation
#include "cores/face/PoseEstimator.hpp"

namespace fs = std::filesystem;

// Constants for pose status visualization
constexpr int YAW_TH = 30;
constexpr int ROLL_TH = 25;
constexpr double PITCH_RATIO_LOWER_TH = 0.6;
constexpr double PITCH_RATIO_UPPER_TH = 1.0;

int main(int argc, char *argv[]) {
    // 参数设置 (输入已修改为固定值)
    std::string image_dir      = "/mnt/nfs/perimeter_cpp_v0/G00005/bodies/";
    std::string det_model      = "/mnt/nfs/det_10g_simplified.onnx";
    std::string rec_model      = "/mnt/nfs/w600k_r50_simplified.onnx";
    std::string provider       = "CUDAExecutionProvider";
    bool        show           = false;
    std::string output_file    = "/mnt/nfs/pose_results_cpp.txt";

    if (!fs::is_directory(image_dir)) {
        std::cerr << "错误：找不到目录 " << image_dir << std::endl;
        return 1;
    }

    std::cout << "正在初始化人脸检测器..." << std::endl;
    std::unique_ptr<FaceAnalyzer> face_analyzer;
    try {
        face_analyzer = std::make_unique<FaceAnalyzer>(det_model, rec_model);
        face_analyzer->prepare(provider, 0.5f, cv::Size(640, 640));
        std::cout << "人脸检测器初始化完成。" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "初始化 FaceAnalyzer 失败: " << e.what() << std::endl;
        std::cerr << "请确保 --det_model 和 --rec_model 参数路径正确。" << std::endl;
        return 1;
    }

    // Recursively find all image files
    std::vector<std::string> image_paths;
    const std::vector<std::string> image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    for (const auto &entry: fs::recursive_directory_iterator(image_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            for (const auto &valid_ext: image_exts) {
                if (ext == valid_ext) {
                    image_paths.push_back(entry.path().string());
                    break;
                }
            }
        }
    }
    std::sort(image_paths.begin(), image_paths.end());

    if (image_paths.empty()) {
        std::cout << "在 " << image_dir << " 中未找到任何图片。" << std::endl;
        return 0;
    }

    std::cout << "找到 " << image_paths.size() << " 张图片，开始处理..." << std::endl;
    std::cout << "结果将保存到: " << output_file << std::endl;

    std::ofstream f_out(output_file);
    if (!f_out.is_open()) {
        std::cerr << "错误：无法打开输出文件 " << output_file << std::endl;
        return 1;
    }

    // Write CSV header, same as Python script
    f_out << "ImagePath,FaceIndex,Yaw,Pitch,Roll\n";

    bool should_quit = false;
    for (const auto &img_path: image_paths) {
        std::cout << "\n--- 处理: " << img_path << " ---" << std::endl;
        cv::Mat frame = cv::imread(img_path);
        if (frame.empty()) {
            std::cout << "  -> 无法读取图片。" << std::endl;
            continue;
        }

        std::vector<Face> faces = face_analyzer->detect(frame);

        if (faces.empty()) {
            std::cout << "  -> 未检测到人脸。" << std::endl;
        } else {
            std::cout << "  -> 检测到 " << faces.size() << " 张人脸" << std::endl;
            for (size_t i = 0; i < faces.size(); ++i) {
                auto &face = faces[i];

                // Use the new static method from the PoseEstimator class
                auto pose_result = PoseEstimator::estimate_pose(frame.size(), face.kps);
                if (!pose_result.has_value()) {
                    std::cout << "  人脸 #" << i + 1 << ": 姿态估计失败。" << std::endl;
                    continue;
                }

                double yaw = pose_result.value().yaw;
                double pitch_score = pose_result.value().pitch_score;
                double roll = pose_result.value().roll;
                std::cout << "  人脸 #" << i + 1 << " 姿态: Yaw=" << std::fixed << std::setprecision(2) << yaw
                          << "°, Pitch_Score=" << pitch_score << ", Roll=" << roll << "°" << std::endl;

                // Write to file, strictly matching the Python script's output format (pitch_score, yaw, roll)
                f_out << fs::path(img_path).filename().string() << "," << i + 1 << ","
                      << std::fixed << std::setprecision(4) << pitch_score << ","
                      << yaw << "," << roll << "\n";

                if (show) {
                    cv::Scalar box_color;
                    if (std::abs(yaw) < YAW_TH && std::abs(roll) < ROLL_TH &&
                        pitch_score > PITCH_RATIO_LOWER_TH && pitch_score < PITCH_RATIO_UPPER_TH) {
                        box_color = cv::Scalar(0, 255, 0); // Green (frontal)
                        std::cout << "    -> 判断为: 正脸" << std::endl;
                    } else {
                        box_color = cv::Scalar(0, 255, 255); // Yellow (side/bad pose)
                        std::cout << "    -> 判断为: 侧脸" << std::endl;
                    }

                    // Draw bounding box and keypoints
                    cv::rectangle(frame, face.bbox, box_color, 2);
                    for (const auto &pt: face.kps) {
                        cv::circle(frame, pt, 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
                    }

                    // Draw pose text above the box
                    std::stringstream ss;
                    ss << "Y:" << std::fixed << std::setprecision(1) << yaw
                       << " P_score:" << std::setprecision(2) << pitch_score << " R:" << std::setprecision(1) << roll;
                    cv::Point text_origin(static_cast<int>(face.bbox.x), static_cast<int>(face.bbox.y) - 10);
                    cv::putText(frame, ss.str(), text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0),
                                1);
                }
            }
        }

        if (show) {
            cv::imshow("Pose", frame);
            if ((cv::waitKey(0) & 0xFF) == 'q') {
                should_quit = true;
            }
        }
        if (should_quit) break;
    }

    if (show) {
        cv::destroyAllWindows();
    }

    f_out.close();
    std::cout << "\n--- 处理完成 ---" << std::endl;

    return 0;
}

