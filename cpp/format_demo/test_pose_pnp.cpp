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

namespace fs = std::filesystem;

// Constants for pose status visualization
constexpr int YAW_TH = 30;
constexpr int PITCH_TH = 25;
constexpr int ROLL_TH = 25;

// 3D model points corresponding to ArcFace 5-point landmarks:
// [left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner]
const std::vector<cv::Point3f> OBJECT_POINTS_3D = {
        {-30.0f, 40.0f,  0.0f},   // Left eye
        {30.0f,  40.0f,  0.0f},    // Right eye
        {0.0f,   20.0f,  30.0f},    // Nose tip
        {-25.0f, -20.0f, 0.0f},  // Left mouth corner
        {25.0f,  -20.0f, 0.0f}    // Right mouth corner
};

// Structure to hold calculated pose angles
struct PoseAngles {
    double pitch, yaw, roll;
};

/**
 * @brief Converts a rotation matrix to Euler angles.
 * @param R The rotation matrix.
 * @return A PoseAngles struct containing pitch, yaw, and roll in degrees.
 */
PoseAngles rotationMatrixToEulerAngles(const cv::Mat &R) {
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;

    double x, y, z;
    if (!singular) {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }

    // Return pitch, yaw, roll in degrees
    return {x * 180.0 / CV_PI, y * 180.0 / CV_PI, z * 180.0 / CV_PI};
}

/**
 * @brief Estimates head pose using solvePnP.
 * @param image_size The size of the input image.
 * @param image_pts A vector of 5 2D keypoints.
 * @return An optional containing PoseAngles. Returns std::nullopt on failure.
 */
std::optional<PoseAngles> estimate_pose(cv::Size image_size, const std::vector<cv::Point2f> &image_pts) {
    size_t n_points = image_pts.size();
    if (n_points < 4) {
        return std::nullopt;
    }

    double focal_length = image_size.width;
    cv::Point2d center(image_size.width / 2.0, image_size.height / 2.0);

    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x,
            0, focal_length, center.y,
            0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F); // Assume no lens distortion

    cv::Mat rvec, tvec;
    bool success = false;

    // Select PnP algorithm based on point count, matching the Python script's logic
    if (n_points >= 6) {
        success = cv::solvePnP(OBJECT_POINTS_3D, image_pts, camera_matrix, dist_coeffs, rvec, tvec, false,
                               cv::SOLVEPNP_ITERATIVE);
    } else { // 4-5 points
        success = cv::solvePnP(OBJECT_POINTS_3D, image_pts, camera_matrix, dist_coeffs, rvec, tvec, false,
                               cv::SOLVEPNP_EPNP);
    }

    if (!success) {
        return std::nullopt;
    }

    // Refine with Levenberg-Marquardt if we used an approximate method for few points
    if (n_points < 6) {
        cv::solvePnPRefineLM(OBJECT_POINTS_3D, image_pts, camera_matrix, dist_coeffs, rvec, tvec);
    }

    cv::Mat rot_mat;
    cv::Rodrigues(rvec, rot_mat);

    PoseAngles angles = rotationMatrixToEulerAngles(rot_mat);
    // Return as yaw, pitch, roll to match Python script's tuple order
    return PoseAngles{angles.yaw, angles.pitch, angles.roll};
}

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

                auto yaw_pitch_roll = estimate_pose(frame.size(), face.kps);
                if (!yaw_pitch_roll.has_value()) {
                    std::cout << "  人脸 #" << i + 1 << ": 姿态估计失败。" << std::endl;
                    continue;
                }

                double yaw = yaw_pitch_roll.value().yaw;
                double pitch = yaw_pitch_roll.value().pitch;
                double roll = yaw_pitch_roll.value().roll;
                std::cout << "  人脸 #" << i + 1 << " 姿态角: Yaw=" << std::fixed << std::setprecision(2) << yaw
                          << "°, Pitch=" << pitch << "°, Roll=" << roll << "°" << std::endl;

                // Write to file with 4 decimal places for direct comparison
                f_out << fs::path(img_path).filename().string() << "," << i + 1 << ","
                      << std::fixed << std::setprecision(4) << yaw << ","
                      << pitch << "," << roll << "\n";

                if (show) {
                    cv::Scalar box_color;
                    if (std::abs(yaw) < YAW_TH && std::abs(pitch) < PITCH_TH && std::abs(roll) < ROLL_TH) {
                        box_color = cv::Scalar(0, 255, 0); // Green (frontal)
                        std::cout << "    -> 判断为: 正脸" << std::endl;
                    } else {
                        box_color = cv::Scalar(0, 0, 255); // Red (side)
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
                       << " P:" << pitch << " R:" << roll;
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

